#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact signed integrals for polynomial and constant-rational cell maps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from fractions import Fraction
from math import factorial, prod
from typing import Literal

from .._exact_polynomial_kernels import exact_jacobian_integrals
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    compose,
    constant,
    derivative,
    ExactScalar,
    Expression,
    multiply,
    normalize_pyramid_reference_expression,
    Polynomial,
    RationalPolynomial,
    scale,
    subtract,
    sum_polynomials,
)
from ..linalg._exact import determinant_small_algebra


type ReferenceMeasureDomain = Literal["simplex", "box", "prism", "pyramid"]


def map_domain(cell_kind: str, /) -> ReferenceMeasureDomain:
    match cell_kind:
        case "triangle" | "tetrahedron" | "simplex:2" | "simplex:3":
            return "simplex"
        case "quadrilateral" | "hexahedron":
            return "box"
        case "prism":
            return "prism"
        case "pyramid":
            return "pyramid"
        case _:
            raise ValueError(
                f"Unsupported exact reference integration cell {cell_kind!r}."
            )


def _polynomial(value: Expression, dimension: int, /) -> Polynomial:
    if not isinstance(value, RationalPolynomial):
        return value
    denominator = value.denominator
    if set(denominator) != {(0,) * dimension}:
        raise ValueError(
            "Exact mapped integration requires a polynomial or constant denominator."
        )
    scalar = denominator[(0,) * dimension]
    if not scalar:
        raise ValueError("Exact mapped integration encountered a zero denominator.")
    scaled = scale(value.numerator, Fraction(1) / scalar)
    if isinstance(scaled, RationalPolynomial):
        raise AssertionError("Scaling a polynomial numerator must remain polynomial.")
    return scaled


def _pyramid_to_box(value: Polynomial, /) -> Polynomial:
    dimension = 3
    u, v, w = axes(dimension)
    one = constant(1, dimension)
    half = Fraction(1, 2)
    taper = sum_polynomials((one, scale(w, Fraction(-1))), dimension)
    x = sum_polynomials((multiply(u, taper), scale(w, half)), dimension)
    y = sum_polynomials((multiply(v, taper), scale(w, half)), dimension)
    coordinates = (x, y, w)
    polynomial = compose(value, coordinates)
    if not isinstance(polynomial, dict):
        raise AssertionError("Polynomial composition must remain polynomial.")
    weight = multiply(taper, taper)
    result = multiply(polynomial, weight)
    if not isinstance(result, dict):
        raise AssertionError("Pyramid integration weight must remain polynomial.")
    return result


def integrate(
    value: Expression,
    domain: ReferenceMeasureDomain,
    dimension: int,
    /,
) -> ExactScalar:
    polynomial = _polynomial(value, dimension)
    if dimension not in (2, 3) or any(
        len(exponent) != dimension for exponent in polynomial
    ):
        raise ValueError(
            "Exact mapped integration requires a two- or three-dimensional polynomial."
        )
    if domain == "simplex" and dimension not in (2, 3):
        raise ValueError("Simplex integration requires dimension two or three.")
    if domain == "prism" and dimension != 3:
        raise ValueError("Prism integration requires dimension three.")
    if domain == "pyramid":
        if dimension != 3:
            raise ValueError("Pyramid integration requires dimension three.")
        polynomial = _pyramid_to_box(polynomial)
        domain = "box"
    total: ExactScalar = Fraction(0)
    for exponent, coefficient in polynomial.items():
        if domain == "simplex":
            moment = Fraction(
                prod(factorial(power) for power in exponent),
                factorial(dimension + sum(exponent)),
            )
        elif domain == "box":
            moment = prod(
                (Fraction(1, power + 1) for power in exponent), start=Fraction(1)
            )
        else:
            first, second, third = exponent
            moment = Fraction(
                factorial(first) * factorial(second), factorial(first + second + 2)
            )
            moment *= Fraction(1, third + 1)
        total += coefficient * moment
    return total


def _prepare_jacobian(
    coordinates: tuple[Expression, ...], domain: ReferenceMeasureDomain, /
) -> tuple[tuple[tuple[Expression, ...], ...], tuple[tuple[Polynomial, ...], ...] | None]:
    dimension = len(coordinates)
    if dimension not in (2, 3):
        raise ValueError("Mapped Jacobian integration requires a 2-D or 3-D map.")
    if domain == "pyramid":
        # The collapsed-cube pullback includes the collapse Jacobian.
        coordinates = tuple(
            normalize_pyramid_reference_expression(value).expression
            for value in coordinates
        )
    jacobian = tuple(
        tuple(derivative(value, axis) for axis in range(dimension))
        for value in coordinates
    )
    if domain == "pyramid" or (domain == "prism" and dimension != 3):
        return jacobian, None
    polynomials: list[Polynomial] = []
    for value in coordinates:
        if isinstance(value, RationalPolynomial):
            denominator = value.denominator
            if set(denominator) != {(0,) * dimension}:
                return jacobian, None
            scalar = denominator[(0,) * dimension]
            if not isinstance(scalar, (Fraction, int)) or not scalar:
                return jacobian, None
            if not all(
                isinstance(coefficient, (Fraction, int))
                for coefficient in value.numerator.values()
            ):
                return jacobian, None
            polynomial = _polynomial(value, dimension)
        else:
            polynomial = value
        if any(
            len(exponent) != dimension or not isinstance(coefficient, (Fraction, int))
            for exponent, coefficient in polynomial.items()
        ):
            return jacobian, None
        polynomials.append(polynomial)
    rows: list[tuple[Polynomial, ...]] = []
    for coordinate, polynomial, row in zip(
        coordinates, polynomials, jacobian, strict=True
    ):
        if isinstance(coordinate, RationalPolynomial):
            rows.append(tuple(derivative(polynomial, axis) for axis in range(dimension)))
        else:
            polynomial_row: list[Polynomial] = []
            for value in row:
                if not isinstance(value, dict):
                    raise AssertionError("Polynomial derivatives must remain polynomial.")
                polynomial_row.append(value)
            rows.append(tuple(polynomial_row))
    return jacobian, tuple(rows)


def integrate_jacobians(
    maps: Iterable[tuple[Expression, ...]],
    cell_kind: str,
    /,
) -> Iterator[ExactScalar]:
    """Integrate compatible maps, yielding results and errors in cell order."""
    domain = map_domain(cell_kind)
    prepared: list[tuple[tuple[Expression, ...], ...]] = []
    candidates: dict[int, list[tuple[int, tuple[tuple[Polynomial, ...], ...]]]] = {}
    preparation_error: Exception | None = None
    try:
        for coordinates in maps:
            jacobian, polynomial = _prepare_jacobian(coordinates, domain)
            index = len(prepared)
            prepared.append(jacobian)
            if polynomial is not None:
                candidates.setdefault(len(jacobian), []).append((index, polynomial))
    except Exception as error:
        # A later cell must not mask an earlier orientation or integration error.
        preparation_error = error

    native: dict[int, Fraction] = {}
    if domain != "pyramid":
        for dimension, group in candidates.items():
            results = exact_jacobian_integrals(
                [jacobian for _, jacobian in group], domain, dimension
            )
            native.update(
                (index, result)
                for (index, _), result in zip(group, results, strict=True)
                if result is not None
            )
    for index, jacobian in enumerate(prepared):
        if index in native:
            yield native[index]
        else:
            determinant = determinant_small_algebra(
                jacobian, add=add, subtract=subtract, multiply=multiply
            )
            yield integrate(
                determinant, "box" if domain == "pyramid" else domain, len(jacobian)
            )
    if preparation_error is not None:
        raise preparation_error


def integrate_jacobian(
    coordinates: tuple[Expression, ...],
    cell_kind: str,
    /,
) -> ExactScalar:
    if len(coordinates) not in (2, 3):
        raise ValueError("Mapped Jacobian integration requires a 2-D or 3-D map.")
    return next(integrate_jacobians((coordinates,), cell_kind))


__all__ = [
    "ReferenceMeasureDomain",
    "integrate",
    "integrate_jacobians",
    "integrate_jacobian",
    "map_domain",
]
