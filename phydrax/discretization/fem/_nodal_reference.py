#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact source expressions for scalar nodal reference functionals.

The JAX arrays owned by nodal families are execution projections.  This module
retains their mathematical node definitions independently, including the
Legendre--Lobatto roots used by simplex warp-and-blend nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import product
from typing import Any, cast, Sequence

import numpy as np

from ..._polynomial._orthogonal import _legendre_coefficients
from ...algebraic._exact_univariate import _derivative, isolate_real_roots
from .._exact_algebraic_geometry import ExactScalar, RadicalElement


@dataclass(frozen=True, slots=True)
class LegendreLobattoRoot:
    """One ordered real root of the exact derivative of a Legendre polynomial."""

    order: int
    index: int
    polynomial: tuple[Fraction, ...]
    lower: Fraction
    upper: Fraction

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "legendre-lobatto-interior-root",
            self.order,
            self.index,
            tuple((value.numerator, value.denominator) for value in self.polynomial),
        )

    @property
    def exact_value(self) -> ExactScalar | None:
        if self.lower == self.upper:
            return self.lower
        polynomial = self.polynomial
        for root in _lobatto_roots(self.order):
            if root.lower == root.upper:
                polynomial = _deflate_rational_root(polynomial, root.lower)
        if len(polynomial) != 3:
            return None
        constant_term, linear_term, quadratic_term = polynomial
        discriminant = linear_term**2 - 4 * quadratic_term * constant_term
        if discriminant < 0:
            return None
        radical = RadicalElement.sqrt(discriminant)
        denominator = 2 * quadratic_term
        candidates = (
            (-linear_term - radical) / denominator,
            (-linear_term + radical) / denominator,
        )
        return next(
            (
                candidate
                for candidate in candidates
                if self.lower <= candidate <= self.upper
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class ExactNodalExpression:
    """Exact formula for one reference-node coordinate, with optional ExactScalar value."""

    operation: str
    arguments: tuple[object, ...]

    @property
    def identity(self) -> tuple[object, ...]:
        if self.operation == "constant":
            value = self.arguments[0]
            if isinstance(value, Fraction):
                return ("rational", value.numerator, value.denominator)
            assert isinstance(value, RadicalElement)
            return (
                "radical",
                tuple(
                    (radicand, coefficient.numerator, coefficient.denominator)
                    for radicand, coefficient in value.terms
                ),
            )
        if self.operation == "lobatto-unit":
            exact = self.exact_value
            return (
                constant(exact).identity
                if exact is not None
                else (self.operation, *self.arguments)
            )
        if self.operation == "root":
            root = self.arguments[0]
            assert isinstance(root, LegendreLobattoRoot)
            return root.identity
        return (
            self.operation,
            *(
                argument.identity
                if isinstance(argument, ExactNodalExpression)
                else argument
                for argument in self.arguments
            ),
        )

    @property
    def exact_value(self) -> ExactScalar | None:
        if self.operation == "constant":
            value = self.arguments[0]
            assert isinstance(value, (Fraction, RadicalElement))
            return value
        if self.operation == "root":
            root = self.arguments[0]
            assert isinstance(root, LegendreLobattoRoot)
            return root.exact_value
        if self.operation == "lobatto-unit":
            order, index = self.arguments
            assert isinstance(order, int) and isinstance(index, int)
            if order == 0:
                return Fraction(1, 2)
            if index == 0:
                return Fraction(0)
            if index == order:
                return Fraction(1)
            opposite = order - index
            if index == opposite:
                return Fraction(1, 2)
            root = _lobatto_root(order, min(index, opposite) - 1)
            value = root.exact_value
            if value is None:
                return None
            lower_unit = (Fraction(1) + value) / 2
            return lower_unit if index < opposite else Fraction(1) - lower_unit
        if self.operation == "power":
            exponent = self.arguments[1]
            assert isinstance(exponent, int)
            value = self.arguments[0]
            assert isinstance(value, ExactNodalExpression)
            exact = value.exact_value
            return None if exact is None else exact**exponent
        values = tuple(
            cast(ExactNodalExpression, argument).exact_value
            for argument in self.arguments
        )
        if any(value is None for value in values):
            return None
        first = values[0]
        assert first is not None
        if self.operation == "add":
            result: ExactScalar = Fraction(0)
            for value in values:
                assert value is not None
                result = result + value
            return result
        if self.operation == "multiply":
            result = Fraction(1)
            for value in values:
                assert value is not None
                result = result * value
            return result
        if self.operation == "divide":
            denominator = values[1]
            assert denominator is not None
            return first / denominator
        raise ValueError(f"Unknown exact nodal expression {self.operation!r}.")

    def interval(self, /, *, bits: int = 192) -> tuple[Fraction, Fraction]:
        """Return an outward rational interval using exact root isolators."""
        if self.operation == "constant":
            value = self.arguments[0]
            assert isinstance(value, (Fraction, RadicalElement))
            return (
                value.interval(bits=bits)
                if isinstance(value, RadicalElement)
                else (value, value)
            )
        if self.operation == "root":
            root = self.arguments[0]
            assert isinstance(root, LegendreLobattoRoot)
            return root.lower, root.upper
        if self.operation == "lobatto-unit":
            order, index = self.arguments
            assert isinstance(order, int) and isinstance(index, int)
            exact = self.exact_value
            if exact is not None:
                return _scalar_interval(exact, bits)
            opposite = order - index
            root = _lobatto_root(order, min(index, opposite) - 1)
            lo, hi = root.lower, root.upper
            if index < opposite:
                return (Fraction(1) + lo) / 2, (Fraction(1) + hi) / 2
            return (Fraction(1) - hi) / 2, (Fraction(1) - lo) / 2
        if self.operation == "power":
            base = self.arguments[0]
            assert isinstance(base, ExactNodalExpression)
            intervals = (base.interval(bits=bits),)
        else:
            intervals = tuple(
                cast(ExactNodalExpression, argument).interval(bits=bits)
                for argument in self.arguments
            )
        if self.operation == "add":
            return (
                sum((lower for lower, _ in intervals), Fraction(0)),
                sum((upper for _, upper in intervals), Fraction(0)),
            )
        if self.operation == "multiply":
            lower = upper = Fraction(1)
            for next_lower, next_upper in intervals:
                values = (
                    lower * next_lower,
                    lower * next_upper,
                    upper * next_lower,
                    upper * next_upper,
                )
                lower, upper = min(values), max(values)
            return lower, upper
        if self.operation == "divide":
            (lower, upper), (denominator_lower, denominator_upper) = intervals
            if denominator_lower <= 0 <= denominator_upper:
                exact = self.exact_value
                if exact is not None:
                    return _scalar_interval(exact, bits)
                raise ArithmeticError("Exact nodal interval division crosses zero.")
            reciprocal = (1 / denominator_upper, 1 / denominator_lower)
            values = (
                lower * reciprocal[0],
                lower * reciprocal[1],
                upper * reciprocal[0],
                upper * reciprocal[1],
            )
            return min(values), max(values)
        if self.operation == "power":
            exponent = self.arguments[1]
            assert isinstance(exponent, int)
            lower, upper = intervals[0]
            if exponent < 0:
                positive = ExactNodalExpression(
                    "power", (self.arguments[0], -exponent)
                ).interval(bits=bits)
                if positive[0] <= 0 <= positive[1]:
                    raise ArithmeticError("Exact nodal interval power crosses zero.")
                return 1 / positive[1], 1 / positive[0]
            if exponent == 0:
                return Fraction(1), Fraction(1)
            if exponent % 2 == 0 and lower <= 0 <= upper:
                return Fraction(0), max(abs(lower), abs(upper)) ** exponent
            values = (lower**exponent, upper**exponent)
            return min(values), max(values)
        raise ValueError(f"Unknown exact nodal expression {self.operation!r}.")


def constant(value: ExactScalar | int, /) -> ExactNodalExpression:
    if isinstance(value, RadicalElement):
        if not value.terms:
            value = Fraction(0)
        elif len(value.terms) == 1 and value.terms[0][0] == 1:
            value = value.terms[0][1]
    scalar = value if isinstance(value, (Fraction, RadicalElement)) else Fraction(value)
    return ExactNodalExpression("constant", (scalar,))


def _scalar_interval(value: ExactScalar, bits: int, /) -> tuple[Fraction, Fraction]:
    return (
        value.interval(bits=bits) if isinstance(value, RadicalElement) else (value, value)
    )


def _as_expression(
    value: ExactNodalExpression | ExactScalar | int, /
) -> ExactNodalExpression:
    return value if isinstance(value, ExactNodalExpression) else constant(value)


def _sort_key(value: ExactNodalExpression, /) -> str:
    return repr(value.identity)


def add(*values: ExactNodalExpression | ExactScalar | int) -> ExactNodalExpression:
    flattened: list[ExactNodalExpression] = []
    scalar: ExactScalar = Fraction(0)
    for raw in values:
        value = _as_expression(raw)
        if value.operation == "add":
            children = value.arguments
        else:
            children = (value,)
        for child in children:
            assert isinstance(child, ExactNodalExpression)
            exact = child.exact_value
            if exact is not None:
                scalar = scalar + exact
            else:
                flattened.append(child)
    if scalar:
        flattened.append(constant(scalar))
    if not flattened:
        return constant(0)
    if len(flattened) == 1:
        return flattened[0]
    flattened.sort(key=_sort_key)
    return ExactNodalExpression("add", tuple(flattened))


def negate(value: ExactNodalExpression | ExactScalar | int, /) -> ExactNodalExpression:
    return multiply(-1, _as_expression(value))


def subtract(
    first: ExactNodalExpression | ExactScalar | int,
    second: ExactNodalExpression | ExactScalar | int,
    /,
) -> ExactNodalExpression:
    return add(first, negate(second))


def one_minus(value: ExactNodalExpression | ExactScalar | int, /) -> ExactNodalExpression:
    value_ = _as_expression(value)
    if value_.operation == "lobatto-unit":
        order, index = value_.arguments
        assert isinstance(order, int) and isinstance(index, int)
        return lobatto_unit(order, order - index)
    return subtract(1, value_)


def multiply(*values: ExactNodalExpression | ExactScalar | int) -> ExactNodalExpression:
    flattened: list[ExactNodalExpression] = []
    scalar: ExactScalar = Fraction(1)
    for raw in values:
        value = _as_expression(raw)
        if value.operation == "multiply":
            children = value.arguments
        else:
            children = (value,)
        for child in children:
            assert isinstance(child, ExactNodalExpression)
            exact = child.exact_value
            if exact is not None:
                scalar = scalar * exact
            else:
                flattened.append(child)
    if not scalar:
        return constant(0)
    if scalar != 1:
        flattened.append(constant(scalar))
    if not flattened:
        return constant(1)
    if len(flattened) == 1:
        return flattened[0]
    flattened.sort(key=_sort_key)
    return ExactNodalExpression("multiply", tuple(flattened))


def divide(
    numerator: ExactNodalExpression | ExactScalar | int,
    denominator: ExactNodalExpression | ExactScalar | int,
    /,
) -> ExactNodalExpression:
    numerator_, denominator_ = _as_expression(numerator), _as_expression(denominator)
    if denominator_.exact_value == 1:
        return numerator_
    if numerator_.exact_value is not None and denominator_.exact_value is not None:
        return constant(numerator_.exact_value / denominator_.exact_value)
    return ExactNodalExpression("divide", (numerator_, denominator_))


def power(
    value: ExactNodalExpression | ExactScalar | int, exponent: int, /
) -> ExactNodalExpression:
    value_ = _as_expression(value)
    if exponent < 0:
        return divide(1, power(value_, -exponent))
    if exponent == 0:
        return constant(1)
    exact = value_.exact_value
    if exact is not None:
        return constant(exact**exponent)
    if exponent == 1:
        return value_
    return ExactNodalExpression("power", (value_, exponent))


def lobatto_unit(order: int, index: int, /) -> ExactNodalExpression:
    if order < 0 or not 0 <= index <= order:
        raise ValueError("Legendre-Lobatto nodal index is outside its order.")
    if order == 0:
        return constant(Fraction(1, 2))
    if index == 0:
        return constant(0)
    if index == order:
        return constant(1)
    if index == order - index:
        return constant(Fraction(1, 2))
    return ExactNodalExpression("lobatto-unit", (order, index))


@cache
def _lobatto_roots(order: int, /) -> tuple[LegendreLobattoRoot, ...]:
    if order < 2:
        return ()
    polynomial = _derivative(_legendre_coefficients(order))
    isolated = isolate_real_roots(polynomial, tolerance=Fraction(1, 1 << 192))
    if len(isolated) != order - 1:
        raise ArithmeticError("Legendre derivative root isolation lost an interior root.")
    return tuple(
        LegendreLobattoRoot(order, index, polynomial, lower, upper)
        for index, (lower, upper, multiplicity) in enumerate(isolated)
        if multiplicity == 1
    )


def _lobatto_root(order: int, index: int, /) -> LegendreLobattoRoot:
    return _lobatto_roots(order)[index]


def _deflate_rational_root(
    polynomial: tuple[Fraction, ...], root: Fraction, /
) -> tuple[Fraction, ...]:
    if len(polynomial) <= 1:
        raise ArithmeticError("Cannot deflate a constant polynomial.")
    degree = len(polynomial) - 1
    quotient = [Fraction(0)] * degree
    quotient[-1] = polynomial[-1]
    for index in range(degree - 1, 0, -1):
        quotient[index - 1] = polynomial[index] + root * quotient[index]
    if polynomial[0] + root * quotient[0]:
        raise ArithmeticError(
            "Isolated rational Lobatto root does not divide its polynomial."
        )
    while len(quotient) > 1 and not quotient[-1]:
        quotient.pop()
    return tuple(quotient)


def _rational_value(value: ExactNodalExpression, /) -> Fraction:
    exact = value.exact_value
    if isinstance(exact, Fraction):
        return exact
    if isinstance(exact, RadicalElement) and (
        not exact.terms or (len(exact.terms) == 1 and exact.terms[0][0] == 1)
    ):
        return Fraction(0) if not exact.terms else exact.terms[0][1]
    raise ArithmeticError("A canonical warp interpolation argument is not rational.")


def _sqrt_rational(value: ExactNodalExpression, /) -> ExactNodalExpression:
    scalar = value.exact_value
    if isinstance(scalar, RadicalElement):
        if scalar.terms and not (len(scalar.terms) == 1 and scalar.terms[0][0] == 1):
            raise ArithmeticError("Warp-and-blend tangent norm is not rational.")
        scalar = Fraction(0) if not scalar.terms else scalar.terms[0][1]
    if not isinstance(scalar, Fraction) or scalar < 0:
        raise ArithmeticError(
            "Warp-and-blend tangent norm is not a nonnegative rational."
        )
    return constant(RadicalElement.sqrt(scalar))


def _alpha_fraction(dimension: int, order: int, /) -> Fraction:
    from ._high_order import _ALPHA_OPT_2D, _ALPHA_OPT_3D

    values = _ALPHA_OPT_2D if dimension == 2 else _ALPHA_OPT_3D
    if order <= len(values):
        return Fraction(str(values[order - 1]))
    return Fraction(5, 3) if dimension == 2 else Fraction(1)


def _equilateral_vertices(
    dimension: int, /
) -> tuple[tuple[ExactNodalExpression, ...], ...]:
    root_three = _sqrt_rational(constant(Fraction(1, 3)))
    if dimension == 2:
        return (
            (constant(-1), negate(root_three)),
            (constant(1), negate(root_three)),
            (constant(0), multiply(2, root_three)),
        )
    root_six = _sqrt_rational(constant(Fraction(1, 6)))
    return (
        (constant(-1), negate(root_three), negate(root_six)),
        (constant(1), negate(root_three), negate(root_six)),
        (constant(0), multiply(2, root_three), negate(root_six)),
        (constant(0), constant(0), multiply(3, root_six)),
    )


def _equilateral_to_unit(
    points: tuple[tuple[ExactNodalExpression, ...], ...], dimension: int, /
) -> tuple[tuple[ExactNodalExpression, ...], ...]:
    root_three = _sqrt_rational(constant(Fraction(1, 3)))
    if dimension == 2:
        matrix = (
            (constant(1), negate(root_three)),
            (constant(0), multiply(2, root_three)),
        )
        offset = (constant(Fraction(-1, 3)), constant(Fraction(-1, 3)))
    else:
        root_six = _sqrt_rational(constant(Fraction(1, 6)))
        matrix = (
            (constant(1), negate(root_three), negate(root_six)),
            (constant(0), multiply(2, root_three), negate(root_six)),
            (constant(0), constant(0), multiply(3, root_six)),
        )
        offset = (constant(Fraction(-1, 2)),) * 3
    return tuple(
        tuple(
            add(
                offset[row],
                *(multiply(matrix[row][axis], point[axis]) for axis in range(dimension)),
            )
            for row in range(dimension)
        )
        for point in points
    )


def _dot(
    first: Sequence[ExactNodalExpression],
    second: Sequence[ExactNodalExpression],
    /,
) -> ExactNodalExpression:
    return add(
        *(multiply(left, right) for left, right in zip(first, second, strict=True))
    )


def _warp_factor(order: int, output: ExactNodalExpression, /) -> ExactNodalExpression:
    x = _rational_value(output)
    if order == 0 or x in (Fraction(-1), Fraction(1)):
        return constant(0)
    equispaced = tuple(
        Fraction(-1) + Fraction(2 * index, order) for index in range(order + 1)
    )
    interpolation = []
    for index, node in enumerate(equispaced):
        weight = Fraction(1)
        for other_index, other in enumerate(equispaced):
            if other_index != index:
                weight *= (x - other) / (node - other)
        interpolation.append(weight)
    lobatto = tuple(
        subtract(multiply(2, lobatto_unit(order, index)), 1) for index in range(order + 1)
    )
    warp = add(
        *(
            multiply(
                interpolation[index],
                subtract(lobatto[index], equispaced[index]),
            )
            for index in range(order + 1)
        )
    )
    return divide(warp, 1 - x * x)


def _equilateral_shift_2d(
    order: int,
    barycentric: tuple[ExactNodalExpression, ...],
    alpha: Fraction,
    /,
) -> tuple[ExactNodalExpression, ExactNodalExpression]:
    vertices = _equilateral_vertices(2)
    result = [constant(0), constant(0)]
    for first in range(3):
        second, third = tuple(index for index in range(3) if index != first)
        blend = multiply(4, barycentric[second], barycentric[third])
        warp = multiply(
            blend,
            _warp_factor(order, subtract(barycentric[second], barycentric[third])),
            add(1, power(multiply(alpha, barycentric[first]), 2)),
        )
        tangent = tuple(
            subtract(vertices[second][axis], vertices[third][axis]) for axis in range(2)
        )
        norm = _sqrt_rational(_dot(tangent, tangent))
        tangent = tuple(divide(value, norm) for value in tangent)
        for axis in range(2):
            result[axis] = add(result[axis], multiply(tangent[axis], warp))
    return result[0], result[1]


@cache
def _warp_and_blend_barycentric(
    order: int,
    dimension: int,
    lattice: tuple[int, ...],
    /,
) -> tuple[ExactNodalExpression, ...]:
    if order == 0:
        return (constant(Fraction(1, dimension + 1)),) * (dimension + 1)
    integer_barycentric = (order - sum(lattice), *lattice)
    if order == 1:
        return tuple(constant(int(index == order)) for index in integer_barycentric)
    barycentric = tuple(constant(Fraction(value, order)) for value in integer_barycentric)
    vertices = _equilateral_vertices(dimension)
    equilateral = tuple(
        add(
            *(
                multiply(vertices[vertex][axis], barycentric[vertex])
                for vertex in range(dimension + 1)
            )
        )
        for axis in range(dimension)
    )
    alpha = _alpha_fraction(dimension, order)
    if dimension == 2:
        shifted = _equilateral_shift_2d(order, barycentric, alpha)
        warped = tuple(add(equilateral[axis], shifted[axis]) for axis in range(2))
    elif dimension == 3:
        shift = [constant(0), constant(0), constant(0)]
        for first, second, third, fourth, vertex_step in (
            (0, 1, 2, 3, -1),
            (1, 2, 3, 0, -1),
            (2, 3, 0, 1, -1),
            (3, 0, 1, 2, -1),
        ):
            vertex_two, vertex_three, vertex_four = (
                (first + vertex_step * index) % 4 for index in range(1, 4)
            )
            tangent_one = tuple(
                subtract(vertices[vertex_three][axis], vertices[vertex_four][axis])
                for axis in range(3)
            )
            tangent_one_norm = _sqrt_rational(_dot(tangent_one, tangent_one))
            tangent_one = tuple(divide(value, tangent_one_norm) for value in tangent_one)
            tangent_two_raw = tuple(
                subtract(vertices[vertex_two][axis], vertices[vertex_three][axis])
                for axis in range(3)
            )
            tangent_projection = _dot(tangent_one, tangent_two_raw)
            tangent_two = tuple(
                subtract(
                    tangent_two_raw[axis], multiply(tangent_projection, tangent_one[axis])
                )
                for axis in range(3)
            )
            tangent_two_norm = _sqrt_rational(_dot(tangent_two, tangent_two))
            tangent_two = tuple(divide(value, tangent_two_norm) for value in tangent_two)

            face_barycentric = tuple(
                barycentric[index] for index in (second, third, fourth)
            )
            warp_one, warp_two = _equilateral_shift_2d(order, face_barycentric, alpha)
            first_barycentric = barycentric[first]
            second_barycentric, third_barycentric, fourth_barycentric = face_barycentric
            blend = multiply(second_barycentric, third_barycentric, fourth_barycentric)
            denominator = multiply(
                add(second_barycentric, multiply(Fraction(1, 2), first_barycentric)),
                add(third_barycentric, multiply(Fraction(1, 2), first_barycentric)),
                add(fourth_barycentric, multiply(Fraction(1, 2), first_barycentric)),
            )
            exact_denominator = denominator.exact_value
            if exact_denominator is None:
                raise ArithmeticError("Simplex lattice denominator is not rational.")
            if exact_denominator > 0:
                blend = divide(
                    multiply(
                        add(1, power(multiply(alpha, first_barycentric), 2)),
                        blend,
                    ),
                    denominator,
                )
            if integer_barycentric[first] == 0:
                shift = [
                    add(
                        multiply(warp_one, tangent_one[axis]),
                        multiply(warp_two, tangent_two[axis]),
                    )
                    for axis in range(3)
                ]
            else:
                shift = [
                    add(
                        shift[axis],
                        multiply(blend, warp_one, tangent_one[axis]),
                        multiply(blend, warp_two, tangent_two[axis]),
                    )
                    for axis in range(3)
                ]
        warped = tuple(add(equilateral[axis], shift[axis]) for axis in range(3))
    else:
        raise ValueError("Exact warp-and-blend nodes require dimension two or three.")

    unit = _equilateral_to_unit((warped,), dimension)[0]
    reference = tuple(divide(add(value, 1), 2) for value in unit)
    raw_barycentric = (subtract(1, add(*reference)), *reference)
    support = tuple(index for index, value in enumerate(integer_barycentric) if value > 0)
    if len(support) == 1:
        return tuple(constant(int(index == support[0])) for index in range(dimension + 1))
    if len(support) == 2:
        first, second = support
        parameter_index = integer_barycentric[second]
        edge_nodes = tuple(lobatto_unit(order, index) for index in range(order + 1))
        result = [constant(0)] * (dimension + 1)
        result[first] = edge_nodes[order - parameter_index]
        result[second] = edge_nodes[parameter_index]
        return tuple(result)
    selected = [constant(0)] * (dimension + 1)
    for index in support:
        selected[index] = raw_barycentric[index]
    total = add(*selected)
    return tuple(
        value if value.exact_value == 0 else divide(value, total) for value in selected
    )


@dataclass(frozen=True, slots=True)
class ExactNodalReferenceNode:
    """One point-evaluation functional and its separate execution projection."""

    local_dof: int
    reference_point: tuple[ExactNodalExpression, ...]
    exact_point: tuple[ExactScalar, ...] | None
    entity_dimension: int
    entity_vertices: tuple[int, ...]
    entity_weights: tuple[ExactNodalExpression, ...]
    entity_lattice_index: tuple[int, ...] | None
    entity_lattice_order: int | None
    functional_identity: tuple[object, ...]
    execution_shadow: tuple[float, ...]
    projection_error_bound: tuple[Fraction, ...]


def _simplex_nodes(owner: Any, /) -> tuple[tuple[ExactNodalExpression, ...], ...]:
    order = int(owner.order)
    dimension = len(owner.multiindices[0]) - 1
    return tuple(
        _warp_and_blend_barycentric(order, dimension, tuple(index[1:]))
        for index in owner.multiindices
    )


def _tensor_axis_nodes(owner: Any, /) -> tuple[tuple[ExactNodalExpression, ...], ...]:
    from ._high_order import ReferenceNodalFamily

    if not isinstance(owner, ReferenceNodalFamily):
        raise TypeError("Tensor nodal reference owner is not canonical.")
    result = []
    for order in owner.orders:
        if owner.node_set == "equispaced":
            result.append(
                tuple(
                    constant(Fraction(index, order))
                    if order
                    else constant(Fraction(1, 2))
                    for index in range(order + 1)
                )
            )
        else:
            result.append(tuple(lobatto_unit(order, index) for index in range(order + 1)))
    return tuple(result)


def _reference_vertices(
    cell_kind: str, /
) -> tuple[tuple[ExactNodalExpression, ...], ...]:
    from .._reference_cell import reference_cell_topology

    return tuple(
        tuple(constant(Fraction(str(value))) for value in vertex)
        for vertex in reference_cell_topology(cell_kind).vertices
    )


def _tensor_weights(
    cell_kind: str,
    point: tuple[ExactNodalExpression, ...],
    /,
) -> tuple[ExactNodalExpression, ...]:
    vertices = _reference_vertices(cell_kind)
    return tuple(
        multiply(
            *(
                point[axis] if vertex[axis].exact_value == 1 else one_minus(point[axis])
                for axis in range(len(point))
            )
        )
        for vertex in vertices
    )


def _pyramid_node_data(
    order: int, /
) -> tuple[
    tuple[tuple[ExactNodalExpression, ...], tuple[ExactNodalExpression, ...]], ...
]:
    if order == 1:
        vertices = _reference_vertices("pyramid")
        result = []
        for index, point in enumerate(vertices):
            weights = tuple(constant(int(index == vertex)) for vertex in range(5))
            result.append((point, weights))
        return tuple(result)
    result = []
    for layer in range(order + 1):
        height = lobatto_unit(order, layer)
        cross_order = order - layer
        cross_nodes = (
            (constant(Fraction(1, 2)),)
            if cross_order == 0
            else tuple(
                lobatto_unit(cross_order, index) for index in range(cross_order + 1)
            )
        )
        width = one_minus(height)
        for first in cross_nodes:
            for second in cross_nodes:
                point = (
                    add(multiply(width, first), multiply(Fraction(1, 2), height)),
                    add(multiply(width, second), multiply(Fraction(1, 2), height)),
                    height,
                )
                weights = (
                    multiply(width, one_minus(first), one_minus(second)),
                    multiply(width, first, one_minus(second)),
                    multiply(width, first, second),
                    multiply(width, one_minus(first), second),
                    height,
                )
                result.append((point, weights))
    return tuple(result)


type _FamilyNodeRecord = tuple[
    tuple[ExactNodalExpression, ...],
    tuple[ExactNodalExpression, ...],
    tuple[int, ...] | None,
]
type _FamilyNodeData = tuple[_FamilyNodeRecord, ...]


def _reorder_tabulator_free_lagrange_data(
    element: Any, data: _FamilyNodeData, /
) -> _FamilyNodeData:
    if element.family != "Lagrange" or element.tabulator is not None:
        return data
    source_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    if source_nodes.ndim != 2 or source_nodes.shape[0] != len(data):
        raise ValueError(
            "Tabulator-free Lagrange nodes do not match canonical family data."
        )
    source_points = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in source_nodes
    )
    if len(set(source_points)) != len(source_points):
        raise ValueError("Tabulator-free Lagrange reference nodes are not unique.")
    owner_rows: dict[tuple[ExactScalar, ...], int] = {}
    for owner_row, (point, _, _) in enumerate(data):
        exact_values = tuple(value.exact_value for value in point)
        if any(value is None for value in exact_values):
            raise ValueError(
                "Tabulator-free Lagrange nodes must match exact canonical reference points."
            )
        exact_point = tuple(value for value in exact_values if value is not None)
        if len(exact_point) != source_nodes.shape[1] or exact_point in owner_rows:
            raise ValueError(
                "Canonical Lagrange basis nodes are not uniquely identifiable."
            )
        owner_rows[exact_point] = owner_row
    if set(owner_rows) != set(source_points):
        raise ValueError(
            "Tabulator-free Lagrange nodes differ from the canonical basis owner."
        )
    return tuple(data[owner_rows[point]] for point in source_points)


def _family_node_data(
    element: Any,
    owner: Any | None,
    /,
) -> _FamilyNodeData:
    from ._high_order import ReferenceNodalFamily, SimplexNodalFamily
    from ._spectral_hp_completion import HybridReferenceFamily

    if isinstance(owner, SimplexNodalFamily):
        barycentric_nodes = _simplex_nodes(owner)
        data = tuple(
            (
                tuple(node[1:]),
                tuple(node),
                tuple(int(value) for value in owner.multiindices[index]),
            )
            for index, node in enumerate(barycentric_nodes)
        )
        return _reorder_tabulator_free_lagrange_data(element, data)
    if isinstance(owner, ReferenceNodalFamily):
        axis_nodes = _tensor_axis_nodes(owner)
        indices = tuple(product(*(range(order + 1) for order in owner.orders)))
        result = []
        for index in indices:
            point = tuple(axis_nodes[axis][value] for axis, value in enumerate(index))
            result.append((point, _tensor_weights(owner.cell_kind, point), tuple(index)))
        return _reorder_tabulator_free_lagrange_data(element, tuple(result))
    if isinstance(owner, HybridReferenceFamily):
        if owner.cell_kind == "prism":
            triangle = SimplexNodalFamily("triangle", owner.orders[0])
            barycentric_nodes = _simplex_nodes(triangle)
            axial = tuple(
                lobatto_unit(owner.orders[1], index)
                for index in range(owner.orders[1] + 1)
            )
            generated = []
            for tri_node in barycentric_nodes:
                for vertical_index, height in enumerate(axial):
                    point = (*tri_node[1:], height)
                    weights = tuple(
                        multiply(
                            tri_node[vertex],
                            one_minus(height) if vertical == 0 else height,
                        )
                        for vertical in (0, 1)
                        for vertex in range(3)
                    )
                    lattice = (
                        *tuple(
                            int(value)
                            for value in triangle.multiindices[
                                len(generated) // len(axial)
                            ]
                        ),
                        vertical_index,
                    )
                    generated.append((point, weights, lattice))
            return tuple(generated[index] for index in owner.basis_permutation)
        nodes = _pyramid_node_data(owner.degree)
        return tuple((point, weights, None) for point, weights in nodes)
    if (
        element.family
        in (
            "Lagrange",
            "TensorProductLagrange",
            "DiscontinuousLagrange",
        )
        and element.degree == 1
    ):
        vertices = _reference_vertices(element.cell_kind)
        return tuple(
            (
                point,
                tuple(constant(int(index == vertex)) for vertex in range(len(vertices))),
                None,
            )
            for index, point in enumerate(vertices)
        )
    if element.family == "DiscontinuousLagrange" and element.degree == 0:
        centers = {
            "interval": ((Fraction(1, 2),),),
            "triangle": ((Fraction(1, 3), Fraction(1, 3)),),
            "quadrilateral": ((Fraction(1, 2), Fraction(1, 2)),),
            "tetrahedron": ((Fraction(1, 4),) * 3,),
            "hexahedron": ((Fraction(1, 2),) * 3,),
            "prism": ((Fraction(1, 3), Fraction(1, 3), Fraction(1, 2)),),
            "pyramid": ((Fraction(1, 2), Fraction(1, 2), Fraction(1, 4)),),
        }
        points = tuple(
            tuple(constant(value) for value in row) for row in centers[element.cell_kind]
        )
        result = []
        for point in points:
            if element.cell_kind == "pyramid":
                height = point[2]
                width = one_minus(height)
                first = divide(
                    subtract(point[0], multiply(Fraction(1, 2), height)), width
                )
                second = divide(
                    subtract(point[1], multiply(Fraction(1, 2), height)), width
                )
                weights = (
                    multiply(width, one_minus(first), one_minus(second)),
                    multiply(width, first, one_minus(second)),
                    multiply(width, first, second),
                    multiply(width, one_minus(first), second),
                    height,
                )
            elif element.cell_kind == "prism":
                barycentric = (subtract(1, add(*point[:2])), *point[:2])
                weights = tuple(
                    multiply(
                        barycentric[vertex],
                        one_minus(point[2]) if vertical == 0 else point[2],
                    )
                    for vertical in (0, 1)
                    for vertex in range(3)
                )
            elif element.cell_kind in ("triangle", "tetrahedron"):
                weights = (subtract(1, add(*point)), *point)
            else:
                weights = _tensor_weights(element.cell_kind, point)
            result.append((point, weights, None))
        return tuple(result)
    raise ValueError(
        f"Finite-element family {element.family!r} has no canonical scalar nodal owner."
    )


def _owner_for_element(element: Any, /) -> Any | None:
    from ._reference import FiniteElementSpec

    tabulator = element.tabulator
    owner = getattr(tabulator, "__self__", None)
    if isinstance(owner, FiniteElementSpec):
        owner = getattr(owner.tabulator, "__self__", None)
    return owner


def _validated_owner(element: Any, /) -> tuple[Any | None, Any]:
    from ._high_order import ReferenceNodalFamily, SimplexNodalFamily
    from ._reference import discontinuous_element, FiniteElementSpec, lagrange_element
    from ._spectral_hp_completion import HybridReferenceFamily

    source_element = element
    owner = _owner_for_element(element)
    if element.family == "DiscontinuousLagrange":
        if element.degree == 0:
            if (
                element.element_id
                != discontinuous_element(element.cell_kind, 0).element_id
            ):
                raise ValueError(
                    "Degree-zero discontinuous element has no canonical nodal owner."
                )
            return None, None
        source_element = getattr(element.tabulator, "__self__", None)
        if not isinstance(source_element, FiniteElementSpec):
            raise ValueError(
                "Discontinuous nodal tabulator lost its canonical source element."
            )
        canonical = discontinuous_element(element.cell_kind, element.degree)
        canonical_source = getattr(canonical.tabulator, "__self__", None)
        if (
            not isinstance(source_element, FiniteElementSpec)
            or not isinstance(canonical_source, FiniteElementSpec)
            or source_element.element_id != canonical_source.element_id
            or element.element_id != canonical.element_id
            or element.tabulator_id != f"discontinuous:{canonical_source.element_id}"
            or source_element.representation != "point_value"
            or source_element.form_basis is not None
            or source_element.value_spec.proxy != "scalar"
            or source_element.value_spec.form_type.degree != 0
        ):
            raise ValueError("Discontinuous element is not the canonical nodal wrapper.")
        owner = _owner_for_element(source_element)
    if isinstance(owner, SimplexNodalFamily):
        canonical_owner = (
            getattr(source_element.tabulator, "__self__", None)
            if source_element.tabulator is not None
            else None
        )
        if (
            canonical_owner is not owner
            or source_element.family != "SimplexLagrange"
            or source_element.cell_kind != owner.cell_kind
            or source_element.degree != owner.order
            or source_element.tabulator_id != owner.family_id
            or source_element.local_dof_count != len(owner.multiindices)
            or not np.array_equal(
                np.asarray(source_element.reference_nodes), np.asarray(owner.nodes)
            )
            or source_element.representation != "point_value"
            or source_element.form_basis is not None
            or source_element.value_spec.proxy != "scalar"
            or source_element.value_spec.form_type.degree != 0
        ):
            raise ValueError("Simplex nodal element lost its canonical family owner.")
    elif isinstance(owner, ReferenceNodalFamily):
        canonical_owner = getattr(source_element.tabulator, "__self__", None)
        if (
            canonical_owner is not owner
            or source_element.family != "TensorProductLagrange"
            or source_element.cell_kind != owner.cell_kind
            or source_element.degree != max(owner.orders)
            or source_element.representation != "point_value"
            or source_element.form_basis is not None
            or source_element.value_spec.proxy != "scalar"
            or source_element.value_spec.form_type.degree != 0
            or source_element.tabulator_id != owner.family_id
            or not np.array_equal(
                np.asarray(source_element.reference_nodes),
                np.asarray(owner.finite_element().reference_nodes),
            )
        ):
            raise ValueError("Tensor nodal element lost its canonical family owner.")
    elif isinstance(owner, HybridReferenceFamily):
        canonical_owner = getattr(source_element.tabulator, "__self__", None)
        if (
            canonical_owner is not owner
            or source_element.family != "HybridLagrange"
            or source_element.cell_kind != owner.cell_kind
            or source_element.degree != owner.degree
            or source_element.representation != "point_value"
            or source_element.form_basis is not None
            or source_element.value_spec.proxy != "scalar"
            or source_element.value_spec.form_type.degree != 0
            or source_element.tabulator_id != owner.family_id
            or not np.array_equal(
                np.asarray(source_element.reference_nodes), np.asarray(owner.nodes)
            )
        ):
            raise ValueError("Hybrid nodal element lost its canonical family owner.")
    elif owner is None:
        canonical = lagrange_element(element.cell_kind, element.degree)
        if source_element is element and element.element_id != canonical.element_id:
            raise ValueError(
                "Finite element has no admitted canonical scalar nodal owner."
            )
        if (
            element.family == "Lagrange"
            and element.tabulator is None
            and element.degree > 1
        ):
            match element.cell_kind:
                case "triangle" | "tetrahedron" | "simplex:2" | "simplex:3":
                    owner = SimplexNodalFamily(element.cell_kind, element.degree)
                case "interval":
                    owner = ReferenceNodalFamily("interval", element.degree)
                case "quadrilateral":
                    owner = ReferenceNodalFamily("quadrilateral", element.degree)
                case "hexahedron":
                    owner = ReferenceNodalFamily("hexahedron", element.degree)
    else:
        raise ValueError("Finite element tabulator has no canonical scalar nodal owner.")
    return owner, source_element


def finite_element_nodal_reference_data(
    element: Any, /
) -> tuple[ExactNodalReferenceNode, ...]:
    """Return exact point functionals and projections in local DOF order."""
    from ._high_order import SimplexNodalFamily
    from ._reference import FiniteElementSpec
    from ._spectral_hp_completion import HybridReferenceFamily

    if not isinstance(element, FiniteElementSpec):
        raise TypeError("element must be a FiniteElementSpec.")
    if (
        element.representation != "point_value"
        or element.form_basis is not None
        or element.value_spec.proxy != "scalar"
        or element.value_spec.form_type.degree != 0
    ):
        raise ValueError("Exact nodal data require scalar point-evaluation functionals.")
    owner, _source_element = _validated_owner(element)
    family_data = _family_node_data(element, owner)
    if len(family_data) != element.local_dof_count:
        raise ValueError("Exact nodal source data do not align with local DOFs.")
    shadow = np.asarray(element.reference_nodes, dtype=np.float64)
    if shadow.shape != (len(family_data), element.topological_dimension):
        raise ValueError("Exact nodal source data have an incompatible shadow shape.")
    membership: list[tuple[int, tuple[int, ...]]] = [(-1, ())] * len(family_data)
    for dimension, entities in enumerate(element.entity_dofs):
        vertices_by_entity = element.entity_vertices[dimension]
        for entity_index, dofs in enumerate(entities):
            for local_dof in dofs:
                if membership[local_dof][0] >= 0:
                    raise ValueError(
                        "A nodal functional is assigned to multiple entities."
                    )
                membership[local_dof] = (
                    dimension,
                    tuple(vertices_by_entity[entity_index]),
                )
    result: list[ExactNodalReferenceNode] = []
    for local_dof, (point, full_weights, lattice) in enumerate(family_data):
        if len(point) != element.topological_dimension:
            raise ValueError(
                "Exact nodal point dimension differs from its finite element."
            )
        entity_dimension, entity_vertices = membership[local_dof]
        if entity_dimension < 0:
            raise ValueError("A nodal functional has no owning reference entity.")
        if len(full_weights) != len(element.entity_vertices[0]):
            raise ValueError(
                "Exact nodal entity weights do not align with cell vertices."
            )
        entity_weights = tuple(full_weights[vertex] for vertex in entity_vertices)
        entity_lattice = None
        entity_lattice_order = None

        if isinstance(owner, SimplexNodalFamily) and lattice is not None:
            entity_lattice = tuple(lattice[vertex] for vertex in entity_vertices)
            entity_lattice_order = int(owner.order)
        elif (
            isinstance(owner, HybridReferenceFamily)
            and owner.cell_kind == "prism"
            and lattice is not None
            and (
                all(vertex < 3 for vertex in entity_vertices)
                or all(vertex >= 3 for vertex in entity_vertices)
            )
        ):
            entity_lattice = tuple(lattice[vertex % 3] for vertex in entity_vertices)
            entity_lattice_order = int(owner.orders[0])
        exact_point_values = tuple(value.exact_value for value in point)
        exact_point = (
            tuple(value for value in exact_point_values if value is not None)
            if all(value is not None for value in exact_point_values)
            else None
        )
        row_shadow = tuple(float(value) for value in shadow[local_dof])
        errors = []
        for value, projection in zip(point, row_shadow, strict=True):
            try:
                lower, upper = value.interval()
            except ArithmeticError:
                lower, upper = Fraction(0), Fraction(1)
            carrier = Fraction(projection)
            errors.append(max(abs(carrier - lower), abs(upper - carrier)))
        functional_identity = (
            "point-evaluation",
            element.family,
            element.cell_kind,
            element.degree,
            tuple(value.identity for value in point),
        )
        result.append(
            ExactNodalReferenceNode(
                local_dof,
                tuple(point),
                exact_point,
                entity_dimension,
                entity_vertices,
                entity_weights,
                entity_lattice,
                entity_lattice_order,
                functional_identity,
                row_shadow,
                tuple(errors),
            )
        )
    return tuple(result)


def canonical_nodal_entity_identity(
    node: ExactNodalReferenceNode,
    cell_vertex_global_ids: Sequence[int],
    /,
    *,
    cell_global_id: int | None = None,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    """Return a global entity/node key and local-to-canonical vertex action."""
    from .._reference_cell import facet_orientation_actions

    if not isinstance(node, ExactNodalReferenceNode):
        raise TypeError("node must be ExactNodalReferenceNode.")
    vertex_ids = tuple(int(value) for value in cell_vertex_global_ids)
    if any(not 0 <= vertex < len(vertex_ids) for vertex in node.entity_vertices):
        raise ValueError("Nodal entity vertex index is outside its owning cell.")
    local_ids = tuple(vertex_ids[vertex] for vertex in node.entity_vertices)
    if len(set(local_ids)) != len(local_ids):
        raise ValueError("Nodal entity vertices must have distinct global IDs.")
    if node.entity_dimension == 0:
        if len(local_ids) != 1:
            raise ValueError("Vertex nodal identity requires one owning vertex.")
        return ("scalar-nodal-vertex", local_ids[0]), (0,)
    if len(local_ids) == 1:
        actions = ((0,),)
    elif node.entity_dimension == 1:
        actions = ((0, 1), (1, 0))
    else:
        shape = (
            "triangle"
            if len(local_ids) == 3
            else "quadrilateral"
            if len(local_ids) == 4
            else None
        )
        if shape is None:
            actions = (tuple(range(len(local_ids))),)
        else:
            actions = tuple(
                action.permutation for action in facet_orientation_actions(shape)
            )
    canonical_order = min(
        actions, key=lambda action: tuple(local_ids[index] for index in action)
    )
    canonical_ids = tuple(local_ids[index] for index in canonical_order)
    if node.entity_vertices == tuple(range(len(vertex_ids))):
        if cell_global_id is None:
            raise ValueError("Cell-interior nodal identity requires a global cell ID.")
        return (
            (
                "scalar-nodal-cell-functional",
                int(cell_global_id),
                node.functional_identity,
            ),
            tuple(range(len(local_ids))),
        )
    if node.entity_lattice_index is not None and node.entity_dimension > 1:
        canonical_parameters: tuple[object, ...] = tuple(
            node.entity_lattice_index[index] for index in canonical_order
        )
        node_rule = ("simplex-lattice", node.entity_lattice_order)
    else:
        canonical_parameters = tuple(
            node.entity_weights[index].identity for index in canonical_order
        )
        node_rule = ("exact-entity-weights",)
    identity = (
        "scalar-nodal-entity-functional",
        node.entity_dimension,
        canonical_ids,
        node_rule,
        canonical_parameters,
    )
    return identity, canonical_order


__all__ = [
    "ExactNodalExpression",
    "ExactNodalReferenceNode",
    "LegendreLobattoRoot",
    "canonical_nodal_entity_identity",
    "finite_element_nodal_reference_data",
    "lobatto_unit",
]
