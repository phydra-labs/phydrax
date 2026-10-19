#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Polynomial B-Rep deformation and certified displacement feasibility.

Mesh certificate checks
-----------------------
[
  {
    "checks": ["mapped_boundary_injectivity"],
    "owner": "phydrax.geometry.brep._deformation.certify_deformation",
    "premises": [
      {"tag": "X", "statement": "The domain is a closed axis-aligned three-dimensional box with strictly positive side lengths; a polynomial map equals the identity on every boundary face, established by fixing all boundary Bernstein controls at their identity positions."},
      {"tag": "N", "statement": "Exact rational Bernstein coefficients, with subdivision as necessary, establish det J greater than zero everywhere on the closed box."},
      {"tag": "T", "statement": "Brouwer degree is invariant under homotopies fixing the boundary, and each preimage of a regular value contributes the sign of its Jacobian determinant."}
    ],
    "conclusion": "The map is injective on the closed box and maps it onto itself: the identity boundary has degree one, every interior preimage contributes positive one, and the inverse function theorem excludes interior preimages of boundary values.",
    "citation": {"kind": "mathematical", "reference": "Deimling, Nonlinear Functional Analysis (1985)", "locator": "Chapter 1, Brouwer degree: normalization, homotopy invariance and regular-value formula; inverse function theorem", "url": "https://doi.org/10.1007/978-3-662-00547-7"},
    "failure_statuses": ["violated", "unresolved"]
  }
]
End mesh certificate checks

Twist and bend are unsupported. Source coefficients are interpreted as exact rationals;
rounded composed controls retain their outward enclosures.
"""

from __future__ import annotations

from fractions import Fraction
from heapq import heappop, heappush
from itertools import product
from math import comb, prod
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import nonnegative_integer, positive_integer
from ...discretization import _coordinate_enclosure as polynomial
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    reserve_polynomial,
)
from ...discretization._exact_algebraic_geometry import rational_coefficient
from ...typing import checked, ConvertibleToArray, parse
from .._interval_enclosure import _evaluate_device_jaxpr, prepare_interval_function
from .._rational import fraction_down, fraction_up
from ..design._schema import (
    DesignState,
    ParameterBinding,
    ParameterCollector,
    ParameterId,
)
from ._coordinate_request import coordinate_source_request
from ._linked_parameters import (
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
)
from ._model import BRepGeometry, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import (
    _bernstein,
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinSurfacePiece,
    BSplineSurfacePatch,
    CylinderPatch,
    PlanePatch,
)
from ._request_preparations import TrimRequestPreparations


BRepDeformationKind: TypeAlias = Literal["lattice", "taper", "shear"]
BRepDeformationStatus: TypeAlias = Literal[
    "certified", "negative", "unresolved", "unsupported"
]


def _box(value: ConvertibleToArray, /) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[1] <= box[0]):
        raise ValueError(
            f"PolynomialDeformation.box must be finite with positive widths and shape (2, 3); got {box}."
        )
    return box


def identity_lattice(box: ConvertibleToArray, degrees: tuple[int, ...], /) -> Array:
    """Identity Bernstein controls, including the fixed boundary controls."""
    bounds = _box(box)
    if len(degrees) != 3:
        raise ValueError(
            f"identity_lattice.degrees must declare three axes; got {degrees}."
        )
    for degree in degrees:
        positive_integer(degree, "identity_lattice.degree")
    axes = [
        jnp.linspace(bounds[0, axis], bounds[1, axis], degree + 1, dtype=jnp.float64)
        for axis, degree in enumerate(degrees)
    ]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)


def _boundary(shape: tuple[int, ...], /) -> np.ndarray:
    indices = np.indices(shape)
    return np.any(
        np.stack(
            [
                (indices[axis] == 0) | (indices[axis] == size - 1)
                for axis, size in enumerate(shape)
            ]
        ),
        axis=0,
    )


@final
class PolynomialDeformation(StrictModule):
    """A Bernstein lattice with fixed boundary, or a named global polynomial.

    ``taper`` uses ``x_perp -> (1 + rate*(x_axis-origin_axis))*x_perp``
    relative to ``origin``; ``shear`` adds ``rate*(x_driver-origin_driver)``
    to the target coordinate. The supplied box owns the positivity proof.
    """

    box: Array
    control_points: Array
    kind: BRepDeformationKind = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    driver: int = eqx.field(static=True)
    rate: Array
    origin: Array

    def __init__(
        self,
        box: ConvertibleToArray,
        control_points: ConvertibleToArray,
        /,
        *,
        kind: BRepDeformationKind = "lattice",
        axis: int = 2,
        driver: int = 0,
        rate: float = 0.0,
        origin: ConvertibleToArray = (0.0, 0.0, 0.0),
    ) -> None:
        axis = nonnegative_integer(axis, "PolynomialDeformation.axis")
        driver = nonnegative_integer(driver, "PolynomialDeformation.driver")
        bounds = _box(box)
        controls = np.asarray(control_points, dtype=np.float64)
        if (
            controls.ndim != 4
            or controls.shape[-1] != 3
            or min(controls.shape[:3]) < 2
            or not np.all(np.isfinite(controls))
        ):
            raise ValueError(
                f"PolynomialDeformation.control_points must be a finite tensor of shape (nx, ny, nz, 3) with nx,ny,nz >= 2; got {controls.shape}."
            )
        identity = np.asarray(
            identity_lattice(bounds, tuple(size - 1 for size in controls.shape[:3]))
        )
        boundary = _boundary(controls.shape[:3])
        if not np.array_equal(controls[boundary], identity[boundary]):
            raise ValueError(
                "PolynomialDeformation.boundary_controls must equal their identity positions exactly; got displaced boundary controls."
            )
        kind = parse(kind, BRepDeformationKind, "PolynomialDeformation.kind")
        center = np.asarray(origin, dtype=np.float64)
        if (
            axis not in (0, 1, 2)
            or driver not in (0, 1, 2)
            or axis == driver
            or not np.isfinite(rate)
            or center.shape != (3,)
            or not np.all(np.isfinite(center))
        ):
            raise ValueError(
                "PolynomialDeformation.named_parameters must contain distinct coordinate axes and finite rate/origin; got invalid parameters."
            )
        self.box = jnp.asarray(bounds, dtype=jnp.float64)
        self.control_points = jnp.asarray(controls, dtype=jnp.float64)
        self.kind = kind
        self.axis, self.driver = axis, driver
        self.rate = jnp.asarray(rate, dtype=jnp.float64)
        self.origin = jnp.asarray(center, dtype=jnp.float64)

    @classmethod
    def taper(
        cls,
        box: ConvertibleToArray,
        /,
        *,
        axis: int,
        rate: float,
        origin: ConvertibleToArray = (0.0, 0.0, 0.0),
    ) -> PolynomialDeformation:
        """Construct an axial linear taper; certify positive scale before use."""
        axis = nonnegative_integer(axis, "PolynomialDeformation.axis")
        return _named(cls, box, "taper", axis, (axis + 1) % 3, rate, origin)

    @classmethod
    def shear(
        cls,
        box: ConvertibleToArray,
        /,
        *,
        axis: int,
        driver: int,
        rate: float,
        origin: ConvertibleToArray = (0.0, 0.0, 0.0),
    ) -> PolynomialDeformation:
        """Construct a determinant-one shear with an explicit global inverse."""
        return _named(cls, box, "shear", axis, driver, rate, origin)

    @property
    def degree(self) -> int:
        """Declared total degree; tensor lattice degrees add, not multiply."""
        return (
            sum(size - 1 for size in self.control_points.shape[:3])
            if self.kind == "lattice"
            else (2 if self.kind == "taper" else 1)
        )

    def evaluate(self, points: Array, /) -> Array:
        if self.kind == "lattice":
            local = (points - self.box[0]) / (self.box[1] - self.box[0])

            def one(value: Array) -> Array:
                result = self.control_points
                for axis in range(3):
                    weights = _bernstein(value[axis], self.control_points.shape[axis] - 1)
                    result = weights @ result.reshape((weights.size, -1))
                    result = result.reshape(self.control_points.shape[axis + 1 :])
                return result

            mapped = jax.vmap(one)(local.reshape(-1, 3)).reshape(points.shape)
            inside = jnp.all((local >= 0.0) & (local <= 1.0), axis=-1)
            return jnp.where(inside[..., None], mapped, points)
        relative = points - self.origin
        if self.kind == "shear":
            return points.at[..., self.axis].add(self.rate * relative[..., self.driver])
        scale = 1.0 + self.rate * relative[..., self.axis]
        factors = jnp.ones(3, dtype=jnp.float64).at[self.axis].set(0.0)
        return points + relative * (scale - 1.0)[..., None] * factors

    def inverse(
        self, points: Array, /, *, certificate: BRepDeformationCertificate
    ) -> Array:
        """Explicit named-form inverse, bound to this map's certified values."""
        if certificate.status != "certified" or certificate.deformation_id != _identity(
            self
        ):
            raise ValueError(
                "PolynomialDeformation.inverse must receive this map's certified positivity evidence; got absent or mismatched evidence."
            )
        if self.kind == "lattice":
            raise ValueError(
                "PolynomialDeformation.inverse must use a named form; got a lattice."
            )
        relative = points - self.origin
        if self.kind == "shear":
            return points.at[..., self.axis].add(-self.rate * relative[..., self.driver])
        scale = 1.0 + self.rate * relative[..., self.axis]
        relative = eqx.error_if(
            relative,
            jnp.any(
                (points[..., self.axis] < self.box[0, self.axis])
                | (points[..., self.axis] > self.box[1, self.axis])
            ),
            "Polynomial taper inverse lies outside the certified positive-scale interval.",
        )
        factors = jnp.ones_like(points).at[..., self.axis].set(scale)
        return self.origin + relative * factors / scale[..., None]


def _named(
    cls: type[PolynomialDeformation],
    box: ConvertibleToArray,
    kind: BRepDeformationKind,
    axis: int,
    driver: int,
    rate: float,
    origin: ConvertibleToArray,
    /,
) -> PolynomialDeformation:
    return cls(
        box,
        identity_lattice(box, (1, 1, 1)),
        kind=kind,
        axis=axis,
        driver=driver,
        rate=rate,
        origin=origin,
    )


def _identity(value: PolynomialDeformation, /) -> str:
    return canonical_fingerprint(
        {
            "arrays": array_tree_fingerprint(value),
            "kind": value.kind,
            "axis": value.axis,
            "driver": value.driver,
        }
    )


def _basis(
    degrees: tuple[int, ...], budget: CoordinateEnclosureBudget, /
) -> tuple[polynomial.Polynomial, ...]:
    entries = prod((degree + 1) * (degree + 2) // 2 for degree in degrees)
    reserve_polynomial(
        entries, entries, len(degrees), 2 * sum(degrees) + 1, budget=budget
    )
    rows = []
    for indices in product(*(range(degree + 1) for degree in degrees)):
        terms: polynomial.Polynomial = {}
        for exponents in product(
            *(
                range(index, degree + 1)
                for index, degree in zip(indices, degrees, strict=True)
            )
        ):
            coefficient = prod(
                comb(degree, index)
                * comb(degree - index, exponent - index)
                * (-1) ** (exponent - index)
                for degree, index, exponent in zip(
                    degrees, indices, exponents, strict=True
                )
            )
            terms[exponents] = Fraction(coefficient)
        rows.append(terms)
    return tuple(rows)


def _coordinates(
    value: PolynomialDeformation, budget: CoordinateEnclosureBudget, /
) -> tuple[polynomial.Polynomial, ...]:
    bounds = np.asarray(value.box)
    variables = polynomial.axes(3, budget=budget)
    physical = tuple(
        polynomial.add(
            polynomial.constant(Fraction(float(bounds[0, axis])), 3, budget=budget),
            polynomial.scale(
                variables[axis],
                Fraction(float(bounds[1, axis])) - Fraction(float(bounds[0, axis])),
                budget=budget,
            ),
            budget=budget,
        )
        for axis in range(3)
    )
    if value.kind == "lattice":
        basis = _basis(tuple(size - 1 for size in value.control_points.shape[:3]), budget)
        controls = _exact_controls(value, budget).reshape(-1, 3)
        return polynomial.linear_combinations(
            basis,
            tuple(tuple(point[axis] for point in controls) for axis in range(3)),
            budget=budget,
        )
    relative = tuple(
        polynomial.add(
            point,
            polynomial.constant(-Fraction(float(value.origin[axis])), 3, budget=budget),
            budget=budget,
        )
        for axis, point in enumerate(physical)
    )
    rate = Fraction(float(value.rate))
    result = list(physical)
    if value.kind == "shear":
        result[value.axis] = polynomial.add(
            result[value.axis],
            polynomial.scale(relative[value.driver], rate, budget=budget),
            budget=budget,
        )
    else:
        for axis in range(3):
            if axis != value.axis:
                term = polynomial.multiply(
                    relative[axis], relative[value.axis], budget=budget
                )
                result[axis] = polynomial.add(
                    result[axis],
                    polynomial.scale(term, rate, budget=budget),
                    budget=budget,
                )
    return tuple(result)


@final
class BRepDeformationCell(StrictModule):
    """Normalized dyadic rational parameter cell and outward determinant range."""

    box: tuple[tuple[Fraction, ...], tuple[Fraction, ...]] = eqx.field(static=True)
    determinant_range: tuple[float, float] = eqx.field(static=True)
    status: BRepDeformationStatus = eqx.field(static=True)


@final
class BRepDeformationCertificate(StrictModule):
    """Orientation/injectivity outcome, retaining every negative or unresolved cell."""

    status: BRepDeformationStatus = eqx.field(static=True)
    deformation_id: str = eqx.field(static=True)
    cells: tuple[BRepDeformationCell, ...] = eqx.field(static=True)
    reason: str = eqx.field(static=True)


def _cell_coefficients(
    value: polynomial.Polynomial,
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Fraction, ...]:
    variables = polynomial.axes(len(lower), budget=budget)
    arguments = tuple(
        polynomial.add(
            polynomial.constant(first, len(lower), budget=budget),
            polynomial.scale(variable, last - first, budget=budget),
            budget=budget,
        )
        for variable, first, last in zip(variables, lower, upper, strict=True)
    )
    return tuple(
        rational_coefficient(coefficient, role="brep_deformation.cell_coefficient")
        for coefficient in polynomial.bernstein_coefficients(
            polynomial.compose(value, arguments, budget=budget),
            "box",
            len(lower),
            budget=budget,
        )
    )


def _certify_polynomial(
    value: polynomial.Polynomial,
    dimension: int,
    maximum_cells: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[BRepDeformationCell, ...]:
    positive_integer(maximum_cells, "certify_deformation.maximum_cells")
    reserve_polynomial(
        2 * maximum_cells,
        2 * dimension * maximum_cells,
        1,
        maximum_cells + 1,
        budget=budget,
    )
    pending: list[
        tuple[
            Fraction,
            int,
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
        ]
    ] = []
    serial = 0

    def enqueue(lower: tuple[Fraction, ...], upper: tuple[Fraction, ...]) -> None:
        nonlocal serial
        coefficients = _cell_coefficients(value, lower, upper, budget)
        heappush(pending, (min(coefficients), serial, lower, upper, coefficients))
        serial += 1

    enqueue((Fraction(0),) * dimension, (Fraction(1),) * dimension)
    result = []
    negative = False
    while pending:
        _, _, lower, upper, coefficients = heappop(pending)
        low, high = min(coefficients), max(coefficients)
        status: BRepDeformationStatus = (
            "certified" if low > 0 else ("negative" if high < 0 else "unresolved")
        )
        negative |= status == "negative"
        if (
            status == "unresolved"
            and not negative
            and len(result) + len(pending) + 2 <= maximum_cells
        ):
            axis = max(range(dimension), key=lambda index: upper[index] - lower[index])
            middle = (upper[axis] + lower[axis]) / 2
            left, right = list(upper), list(lower)
            left[axis], right[axis] = middle, middle
            enqueue(lower, tuple(left))
            enqueue(tuple(right), upper)
        else:
            result.append(
                BRepDeformationCell(
                    box=(lower, upper),
                    determinant_range=(fraction_down(low), fraction_up(high)),
                    status=status,
                )
            )
    return tuple(sorted(result, key=lambda cell: cell.box))


def _determinant(
    value: PolynomialDeformation, budget: CoordinateEnclosureBudget, /
) -> polynomial.Polynomial:
    determinant = polynomial.determinant_polynomial(
        _coordinates(value, budget), "hexahedron", 3, budget=budget
    )
    if determinant is None:
        raise RuntimeError(
            "PolynomialDeformation.determinant must be polynomial; got a rational expression."
        )
    bounds = np.asarray(value.box)
    volume = prod(
        Fraction(float(bounds[1, axis])) - Fraction(float(bounds[0, axis]))
        for axis in range(3)
    )
    return polynomial.scale(determinant, Fraction(1) / volume, budget=budget)


@checked
def certify_deformation(
    value: PolynomialDeformation,
    /,
    *,
    maximum_cells: int,
    coordinate_budget: CoordinateEnclosureBudget,
) -> BRepDeformationCertificate:
    """Certify strict orientation and global box/named-form injectivity, never sample."""
    cells = _certify_polynomial(
        _determinant(value, coordinate_budget), 3, maximum_cells, coordinate_budget
    )
    status: BRepDeformationStatus = (
        "negative"
        if any(cell.status == "negative" for cell in cells)
        else (
            "unresolved"
            if any(cell.status == "unresolved" for cell in cells)
            else "certified"
        )
    )
    reason = (
        "positive determinant and identity boundary"
        if value.kind == "lattice"
        else "positive determinant and explicit named inverse"
    )
    if status == "negative":
        reason = "negative determinant in a certified cell"
    elif status == "unresolved":
        reason = "determinant sign unresolved within the declared cell budget"
    if value.kind == "taper":
        bounds = np.asarray(value.box)
        scales = tuple(
            Fraction(1)
            + Fraction(float(value.rate))
            * (Fraction(float(endpoint)) - Fraction(float(value.origin[value.axis])))
            for endpoint in bounds[:, value.axis]
        )
        if min(scales) <= 0:
            status, reason = (
                "unsupported",
                "taper scale is not strictly positive on the closed box",
            )
    return BRepDeformationCertificate(
        status=status, deformation_id=_identity(value), cells=cells, reason=reason
    )


def _exact_controls(
    value: PolynomialDeformation, budget: CoordinateEnclosureBudget, /
) -> np.ndarray:
    """Boundary identity is exact rational metadata, not rounded linspace data."""
    reserve_polynomial(
        value.control_points.size,
        value.control_points.size,
        3,
        4300 + max(value.control_points.shape).bit_length(),
        budget=budget,
    )
    points = np.asarray(value.control_points)
    bounds = np.asarray(value.box)
    result = np.asarray(
        [Fraction(float(item)) for item in points.reshape(-1)], dtype=object
    ).reshape(points.shape)
    boundary = _boundary(points.shape[:3])
    for indices in product(*(range(size) for size in points.shape[:3])):
        if boundary[indices]:
            for axis in range(3):
                first, last = (
                    Fraction(float(bounds[0, axis])),
                    Fraction(float(bounds[1, axis])),
                )
                result[(*indices, axis)] = first + (last - first) * Fraction(
                    indices[axis], points.shape[axis] - 1
                )
    return result


def _spline_homogeneous(
    patch: BSplineSurfacePatch, bounds: np.ndarray, budget: CoordinateEnclosureBudget, /
) -> tuple[np.ndarray, tuple[int, int]]:
    knots = (np.asarray(patch.u_knots), np.asarray(patch.v_knots))
    degrees = patch.u_degree, patch.v_degree
    shape = patch.control_points.shape[:2]
    if shape != tuple(degree + 1 for degree in degrees) or any(
        not np.array_equal(knot, np.repeat(bounds[:, axis], degree + 1))
        for axis, (knot, degree) in enumerate(zip(knots, degrees, strict=True))
    ):
        raise ValueError(
            "deform_brep.patch must be a single clamped rational Bernstein span; got a multi-span or restricted spline."
        )
    reserve_polynomial(prod(shape) * 4, prod(shape) * 4, 2, 4300, budget=budget)
    points, weights = np.asarray(patch.control_points), np.asarray(patch.weights)
    controls = np.empty((*shape, 4), dtype=object)
    for i, j in product(range(shape[0]), range(shape[1])):
        weight = Fraction(float(weights[i, j]))
        if weight <= 0:
            raise ValueError(
                "deform_brep.weights must be positive; got a nonpositive rational weight."
            )
        controls[i, j, 3] = weight
        for axis in range(3):
            controls[i, j, axis] = Fraction(float(points[i, j, axis])) * weight
    return controls, degrees


def _homogeneous(
    patch: AbstractSurfacePatch, bounds: np.ndarray, budget: CoordinateEnclosureBudget, /
) -> tuple[np.ndarray, tuple[int, int]]:
    if isinstance(patch, PlanePatch):
        reserve_polynomial(16, 16, 2, 6450, budget=budget)
        controls = np.ones((2, 2, 4), dtype=object)
        origin, first, second = (
            np.asarray(leaf)
            for leaf in (patch.origin, patch.first_axis, patch.second_axis)
        )
        for i, j, axis in product(range(2), range(2), range(3)):
            controls[i, j, axis] = (
                Fraction(float(origin[axis]))
                + Fraction(float(first[axis])) * Fraction(float(bounds[i, 0]))
                + Fraction(float(second[axis])) * Fraction(float(bounds[j, 1]))
            )
        return controls, (1, 1)
    if isinstance(patch, BSplineSurfacePatch):
        return _spline_homogeneous(patch, bounds, budget)
    if isinstance(patch, BernsteinSurfacePiece):
        if not np.array_equal(
            patch.controls_lower, patch.controls_upper
        ) or not np.array_equal(bounds, np.stack((patch.lower, patch.upper))):
            raise ValueError(
                "deform_brep.patch must contain literal Bernstein coefficients over the full span; got enclosed or restricted coefficients."
            )
        controls = np.asarray(patch.controls)
        if not np.all(controls[..., 3] > 0):
            raise ValueError(
                "deform_brep.weights must be positive; got a nonpositive Bernstein denominator."
            )
        budget.reserve(controls.size, controls.size * 128)
        return np.asarray(
            [Fraction(float(item)) for item in controls.reshape(-1)], dtype=object
        ).reshape(controls.shape), (controls.shape[0] - 1, controls.shape[1] - 1)
    raise ValueError(
        f"deform_brep.patch must be a supported carrier; got {type(patch).__name__}."
    )


def _tensor_polynomials(
    controls: np.ndarray, degrees: tuple[int, int], budget: CoordinateEnclosureBudget, /
) -> tuple[polynomial.Polynomial, ...]:
    basis = _basis(degrees, budget)
    rows = controls.reshape(-1, 4)
    return polynomial.linear_combinations(
        basis, tuple(tuple(row[axis] for row in rows) for axis in range(4)), budget=budget
    )


def _dense_coefficients(
    value: polynomial.Polynomial,
    degrees: tuple[int, int],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Fraction, ...]:
    numerator_bits, denominator_bits = polynomial.coefficient_profile(
        (value,), budget=budget
    )
    count = prod(degree + 1 for degree in degrees)
    bits = (
        numerator_bits
        + denominator_bits
        + 2 * sum(degrees) * max(len(value), 1)
        + max(len(value), 1).bit_length()
    )
    reserve_polynomial(count * len(value), count, 2, bits, budget=budget)
    return tuple(
        sum(
            (
                rational_coefficient(coefficient, role="brep_deformation.coefficient")
                * Fraction(comb(i, alpha[0]), comb(degrees[0], alpha[0]))
                * Fraction(comb(j, alpha[1]), comb(degrees[1], alpha[1]))
                for alpha, coefficient in value.items()
                if alpha[0] <= i and alpha[1] <= j
            ),
            Fraction(0),
        )
        for i, j in product(range(degrees[0] + 1), range(degrees[1] + 1))
    )


def _power(
    value: polynomial.Polynomial, degree: int, budget: CoordinateEnclosureBudget, /
) -> polynomial.Polynomial:
    return polynomial.math_product_polynomials((value,) * degree, 2, budget=budget)


def _compose_homogeneous(
    coordinates: tuple[polynomial.Polynomial, ...],
    source: tuple[polynomial.Polynomial, ...],
    degree: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[polynomial.Polynomial, ...]:
    numerator, weight = source[:3], source[3]
    result = []
    for coordinate in coordinates:
        terms = []
        for alpha, coefficient in coordinate.items():
            factors = tuple(
                _power(item, exponent, budget)
                for item, exponent in zip(numerator, alpha, strict=True)
            ) + (_power(weight, degree - sum(alpha), budget),)
            terms.append(
                polynomial.scale(
                    polynomial.math_product_polynomials(factors, 2, budget=budget),
                    coefficient,
                    budget=budget,
                )
            )
        result.append(polynomial.sum_polynomials(tuple(terms), budget=budget))
    return (*result, _power(weight, degree, budget))


def _rounded(
    values: tuple[Fraction, ...], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([float(item) for item in values], dtype=np.float64),
        np.asarray([fraction_down(item) for item in values], dtype=np.float64),
        np.asarray([fraction_up(item) for item in values], dtype=np.float64),
    )


@final
class _Composition(StrictModule):
    template: BernsteinSurfacePiece
    matrix: Array
    matrix_error: Array
    control_error: Array
    fixed: bool = eqx.field(static=True)

    def realize(self, controls: Array, /) -> BernsteinSurfacePiece:
        if self.fixed:
            return self.template
        points = controls.reshape(-1, 3)
        coordinates = self.matrix @ points
        magnitude = jnp.abs(self.matrix) @ jnp.abs(points)
        error = (
            self.matrix_error @ jnp.abs(points)
            + (jnp.abs(self.matrix) + self.matrix_error) @ self.control_error
            + (4 * self.matrix.shape[1] + 8) * jnp.finfo(jnp.float64).eps * magnitude
        )
        shape = self.template.controls.shape
        output = self.template.controls.at[..., :3].set(
            coordinates.reshape((*shape[:2], 3))
        )
        low = self.template.controls_lower.at[..., :3].set(
            (coordinates - error).reshape((*shape[:2], 3))
        )
        high = self.template.controls_upper.at[..., :3].set(
            (coordinates + error).reshape((*shape[:2], 3))
        )
        return eqx.tree_at(
            lambda value: (value.controls, value.controls_lower, value.controls_upper),
            self.template,
            (output, low, high),
        )


def _normalized_homogeneous(
    value: PolynomialDeformation,
    homogeneous: tuple[polynomial.Polynomial, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[polynomial.Polynomial, ...]:
    bounds = np.asarray(value.box)
    return tuple(
        polynomial.scale(
            polynomial.add(
                item,
                polynomial.scale(
                    homogeneous[3], -Fraction(float(bounds[0, axis])), budget=budget
                ),
                budget=budget,
            ),
            Fraction(1)
            / (Fraction(float(bounds[1, axis])) - Fraction(float(bounds[0, axis]))),
            budget=budget,
        )
        for axis, item in enumerate(homogeneous[:3])
    )


def _lattice_composition(
    value: PolynomialDeformation,
    homogeneous: tuple[polynomial.Polynomial, ...],
    target: tuple[int, int],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[tuple[Fraction, ...], ...], np.ndarray, np.ndarray, np.ndarray]:
    normalized = _normalized_homogeneous(value, homogeneous, budget)
    basis = _basis(tuple(size - 1 for size in value.control_points.shape[:3]), budget)
    composed = _compose_homogeneous(
        basis, (*normalized, homogeneous[3]), value.degree, budget
    )
    columns = tuple(_dense_coefficients(item, target, budget) for item in composed[:-1])
    count = len(columns) * len(columns[0])
    budget.reserve(count, 9 * count * 8 + value.control_points.size * 8)
    triples = tuple(_rounded(column) for column in columns)
    matrix, lower, upper = (
        np.stack([column[index] for column in triples], axis=-1) for index in range(3)
    )
    points = _exact_controls(value, budget).reshape(-1, 3)
    numerator_bits, denominator_bits = polynomial.coefficient_profile(
        composed, budget=budget
    )
    bits = (
        numerator_bits
        + denominator_bits
        + 2 * sum(target) * max(sum(map(len, composed)), 1)
        + 4300
        + len(columns).bit_length()
    )
    reserve_polynomial(count * 3, len(columns[0]) * 3, 2, bits, budget=budget)
    rows = tuple(
        tuple(
            sum(
                (
                    columns[column][row] * points[column, axis]
                    for column in range(points.shape[0])
                ),
                Fraction(0),
            )
            for row in range(len(columns[0]))
        )
        for axis in range(3)
    )
    weight = _dense_coefficients(composed[-1], target, budget)
    nominal = np.asarray(value.control_points).reshape(-1, 3)
    control_error = np.asarray(
        [
            fraction_up(abs(exact - Fraction(float(rounded))))
            for exact, rounded in zip(
                points.reshape(-1), nominal.reshape(-1), strict=True
            )
        ],
        dtype=np.float64,
    ).reshape(nominal.shape)
    return (
        (*rows, weight),
        matrix,
        np.maximum(matrix - lower, upper - matrix),
        control_error,
    )


def _named_composition(
    value: PolynomialDeformation,
    homogeneous: tuple[polynomial.Polynomial, ...],
    target: tuple[int, int],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    variables = (
        *polynomial.axes(3, budget=budget),
        polynomial.constant(1, 3, budget=budget),
    )
    arguments = _normalized_homogeneous(value, variables, budget)
    coordinates = tuple(
        polynomial.compose(item, arguments, budget=budget)
        for item in _coordinates(value, budget)
    )
    composed = _compose_homogeneous(coordinates, homogeneous, value.degree, budget)
    return tuple(_dense_coefficients(item, target, budget) for item in composed)


def _prepare_composition(
    value: PolynomialDeformation,
    patch: AbstractSurfacePatch,
    bounds: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> _Composition:
    source, degrees = _homogeneous(patch, bounds, budget)
    homogeneous = _tensor_polynomials(source, degrees, budget)
    target = (value.degree * degrees[0], value.degree * degrees[1])
    budget.reserve(
        prod(degree + 1 for degree in target) * 4,
        prod(degree + 1 for degree in target) * 1024,
    )
    if value.kind == "lattice":
        rows, matrix, matrix_error, control_error = _lattice_composition(
            value, homogeneous, target, budget
        )
    else:
        rows = _named_composition(value, homogeneous, target, budget)
        matrix = matrix_error = np.zeros((1, 1), dtype=np.float64)
        control_error = np.zeros((1, 3), dtype=np.float64)
    triples = tuple(_rounded(row) for row in rows)
    arrays = tuple(
        np.stack([row[index] for row in triples], axis=-1).reshape(
            (target[0] + 1, target[1] + 1, 4)
        )
        for index in range(3)
    )
    template = BernsteinSurfacePiece(
        arrays[0],
        bounds[0],
        bounds[1],
        controls_lower=arrays[1],
        controls_upper=arrays[2],
    )
    return _Composition(
        template=template,
        matrix=jnp.asarray(matrix, dtype=jnp.float64),
        matrix_error=jnp.asarray(matrix_error, dtype=jnp.float64),
        control_error=jnp.asarray(control_error, dtype=jnp.float64),
        fixed=value.kind != "lattice",
    )


def _patch_location(controls: np.ndarray, box: np.ndarray, /) -> str:
    points = controls[..., :3] / controls[..., 3, None]
    lower, upper = np.min(points, axis=(0, 1)), np.max(points, axis=(0, 1))
    if all(
        lower[axis] >= Fraction(float(box[0, axis]))
        and upper[axis] <= Fraction(float(box[1, axis]))
        for axis in range(3)
    ):
        return "inside"
    if any(
        upper[axis] < Fraction(float(box[0, axis]))
        or lower[axis] > Fraction(float(box[1, axis]))
        for axis in range(3)
    ):
        return "outside"
    return "unresolved"


@final
class TaperedCylinderPatch(AbstractSurfacePatch):
    """Cone expressed by source radius and polynomial taper operands.

    Keeping the radial slope as a product avoids rounding it through
    arctan/tan into the angle representation of an ordinary ConePatch.
    """

    source: CylinderPatch
    deformation: PolynomialDeformation

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.source.periods

    def native_period_symbols(self) -> tuple[str | None, str | None]:
        return self.source.native_period_symbols()

    @property
    def radial_slope(self) -> Array:
        return self.source.radius * self.deformation.rate

    def evaluate(self, parameters: Array, /) -> Array:
        return self.deformation.evaluate(self.source.evaluate(parameters))

    def interval_box(self, box: Array, /) -> Array:
        program = jax.make_jaxpr(self.evaluate)(box[0])
        (result,) = _evaluate_device_jaxpr(program, ((box[0], box[1]),))
        return jnp.stack(result)

    def bounding_box(
        self,
        parameter_box: ConvertibleToArray,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request((self, parameter_box)) as budget:
                return self.bounding_box(
                    parameter_box, preparations=preparations, budget=budget
                )
        domain = self.validate_parameter_box(parameter_box, budget=budget)
        budget.reserve(1, 4096)
        function = prepare_interval_function(self.evaluate, 2, batch_capacity=1)
        low, high = function.evaluate(domain[0][None], domain[1][None])
        return np.stack((low[0], high[0]))


def _taper_cylinder(
    value: PolynomialDeformation, patch: CylinderPatch, /
) -> TaperedCylinderPatch:
    axis = np.asarray(patch.axis)
    expected = np.eye(3, dtype=np.float64)[value.axis]
    center = np.asarray(value.origin)
    origin = np.asarray(patch.origin)
    transverse = [index for index in range(3) if index != value.axis]
    if not np.array_equal(axis, expected) or not np.array_equal(
        origin[transverse], center[transverse]
    ):
        raise ValueError(
            "deform_brep.taper_cylinder must use an exactly aligned cylinder sharing the taper axis; got another axis."
        )
    return TaperedCylinderPatch(source=patch, deformation=value)


@final
class _DeformedCurve(AbstractCurve):
    base: AbstractCurve
    deformation: PolynomialDeformation

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        return self.base.period

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.base.parameter_domain

    def native_period_symbol(self) -> str | None:
        return self.base.native_period_symbol()

    def evaluate(self, parameter: Array, /) -> Array:
        return self.deformation.evaluate(self.base.evaluate(parameter))

    def bounding_box(
        self,
        first: float,
        last: float,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if budget is None:
            with coordinate_source_request((self, first, last)) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        first, last = self.validate_range(first, last)
        budget.reserve(1, 4096 + self.deformation.control_points.size * 128)

        def point(parameter: Array) -> Array:
            return self.evaluate(parameter[0])

        function = prepare_interval_function(point, 1, batch_capacity=1)
        lower, upper = function.evaluate(
            np.asarray([[first]], dtype=np.float64),
            np.asarray([[last]], dtype=np.float64),
        )
        return np.stack((lower[0], upper[0]))


@final
class BRepDeformationMap(AbstractBRepParameterMap):
    """Prepared polynomial carriers for FixedTopologyBRepSource(parameter_map=...).

    Lattice interior control positions are one dynamic design leaf. Boundary
    controls are reconstructed from immutable identity positions, never exposed
    as independently trainable values. Re-certify after a design update before
    publishing that update as an injective represented B-Rep.
    """

    deformation: PolynomialDeformation
    compositions: tuple[_Composition | None, ...]
    patches: tuple[AbstractSurfacePatch, ...]
    model_id: str = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def compile(
        self, context: ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        if model.model_id != self.model_id:
            raise ValueError(
                "BRepDeformationMap.model must match its prepared native model; got another model."
            )
        indices = np.flatnonzero(
            ~_boundary(self.deformation.control_points.shape[:3]).reshape(-1)
        )
        binding = None
        if self.deformation.kind == "lattice":
            binding = context.bind(
                ParameterId(self.feature_id, "control_points"),
                self.deformation.control_points.reshape(-1, 3)[indices],
                role="length",
                unit=context.coordinate_unit,
            )
        return _BoundDeformationMap(
            deformation=self.deformation,
            compositions=self.compositions,
            patches=self.patches,
            binding=binding,
            indices=tuple(indices.tolist()),
            geometry=model.require_geometry(),
            parameter_bounds=model.parameter_bounds,
        )


@final
class _BoundDeformationMap(AbstractBoundBRepParameterMap):
    deformation: PolynomialDeformation
    compositions: tuple[_Composition | None, ...]
    patches: tuple[AbstractSurfacePatch, ...]
    binding: ParameterBinding | None = eqx.field(static=True)
    indices: tuple[int, ...] = eqx.field(static=True)
    geometry: BRepGeometry
    parameter_bounds: Array

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        value = self.deformation
        if self.binding is not None:
            controls = (
                value.control_points.reshape(-1, 3)
                .at[jnp.asarray(self.indices, dtype=jnp.int32)]
                .set(self.binding.read(state))
                .reshape(value.control_points.shape)
            )
            value = eqx.tree_at(lambda item: item.control_points, value, controls)
        patches = tuple(
            patch if composition is None else composition.realize(value.control_points)
            for patch, composition in zip(self.patches, self.compositions, strict=True)
        )
        curves = []
        for curve in self.geometry.curves:
            if not isinstance(curve, AbstractCurve):
                raise ValueError(
                    "BRepDeformationMap.curves must contain explicit native carriers; got an implicit intersection curve."
                )
            curves.append(_DeformedCurve(base=curve, deformation=value))
        geometry = eqx.tree_at(
            lambda item: (item.vertex_points, item.curves),
            self.geometry,
            (value.evaluate(self.geometry.vertex_points), tuple(curves)),
        )
        return BRepParameterGeometry(
            patches=patches, geometry=geometry, parameter_bounds=self.parameter_bounds
        )


@final
class BRepDeformationResult(StrictModule):
    """Explicit construction outcome with polynomial carriers and a linked design map."""

    status: BRepDeformationStatus = eqx.field(static=True)
    certificate: BRepDeformationCertificate
    parameter_map: BRepDeformationMap | None
    reason: str = eqx.field(static=True)
    parameter_range: CertifiedParameterRange | None = None


def _scale_parameter_range(
    upper: Fraction,
    tested_upper: Fraction,
    maximum_scale: float,
    cells: tuple[BRepDeformationCell, ...],
    feasible_cells: tuple[BRepDeformationCell, ...],
    /,
) -> CertifiedParameterRange:
    """Publish the shared interval vocabulary and its determinant evidence."""
    if fraction_down(upper) <= 0.0:
        return unknown_parameter_range(
            "displacement_scale",
            "No positive displacement-scale interval was certified within the declared work budgets.",
        )
    determinant_bounds = (
        min(cell.determinant_range[0] for cell in feasible_cells),
        max(cell.determinant_range[1] for cell in feasible_cells),
    )
    return CertifiedParameterRange(
        "displacement_scale",
        (0.0, fraction_down(upper)),
        end_status=("attained_exactly", "certified_inner_bound"),
        constraints=("parameter_domain", "det_j_sign"),
        evidence=(
            f"Four-dimensional rational Bernstein proof covers every spatial point and every scale in [0,{upper}].",
            f"The searched scale cap is {maximum_scale}; the last tested bracket endpoint is {tested_upper}; no singular endpoint is asserted attained.",
            f"Accepted physical determinant bounds: {determinant_bounds}.",
            f"Last tested cells: {tuple((cell.box, cell.determinant_range, cell.status) for cell in cells)}.",
        ),
    )


@checked
def deform_brep(
    model: BRepModel,
    value: PolynomialDeformation,
    /,
    *,
    maximum_cells: int,
    coordinate_budget: CoordinateEnclosureBudget,
    feature_id: str,
    include_parameter_range: bool = False,
    range_maximum_scale: float = 1.0,
    range_refinements: int = 8,
) -> BRepDeformationResult:
    """Prepare certified polynomial carriers without changing native trims.

    Consume the returned map with ``FixedTopologyBRepSource`` on the source
    model. A crossing or unenclosed patch refuses the entire request; no patch
    is clipped, sampled, or partially deformed.
    The map and box use native definition coordinates. Assembly occurrences
    retain their declared placements; this is a shared definition deformation,
    not a world-space deformation independently materialized for each instance.
    """
    if not feature_id:
        raise ValueError(
            "deform_brep.feature_id must be nonempty; got an empty identity."
        )
    if not isinstance(include_parameter_range, bool):
        raise ValueError(
            "deform_brep.include_parameter_range must be a boolean; got another type."
        )
    unsupported_curves = any(
        not isinstance(curve, AbstractCurve) for curve in model.require_geometry().curves
    )
    certificate = certify_deformation(
        value, maximum_cells=maximum_cells, coordinate_budget=coordinate_budget
    )
    parameter_range = None
    if include_parameter_range or certificate.status != "certified":
        parameter_range = deformation_feasibility_scale(
            value,
            maximum_scale=range_maximum_scale,
            refinements=range_refinements,
            maximum_cells=maximum_cells,
            coordinate_budget=coordinate_budget,
        )
    if unsupported_curves:
        reason = "implicit intersection edge carriers are unsupported"
        return BRepDeformationResult(
            status="unsupported",
            certificate=certificate,
            parameter_map=None,
            reason=reason,
            parameter_range=unknown_parameter_range("displacement_scale", reason),
        )
    if certificate.status != "certified":
        return BRepDeformationResult(
            status=certificate.status,
            certificate=certificate,
            parameter_map=None,
            reason=certificate.reason,
            parameter_range=parameter_range,
        )
    compositions, patches = [], []
    bounds = np.asarray(model.parameter_bounds)
    try:
        for patch, domain in zip(model.patches, bounds, strict=True):
            if isinstance(patch, CylinderPatch) and value.kind == "taper":
                enclosure = patch.bounding_box(domain, budget=coordinate_budget)
                if not np.all(enclosure[0] >= np.asarray(value.box)[0]) or not np.all(
                    enclosure[1] <= np.asarray(value.box)[1]
                ):
                    raise ValueError(
                        "deform_brep.patch must lie wholly inside the taper proof box; got an unenclosed cylinder."
                    )
                compositions.append(None)
                patches.append(_taper_cylinder(value, patch))
                continue
            controls, _ = _homogeneous(patch, domain, coordinate_budget)
            location = _patch_location(controls, np.asarray(value.box))
            if location == "outside" and value.kind == "lattice":
                compositions.append(None)
                patches.append(patch)
            elif location != "inside":
                raise ValueError(
                    "deform_brep.patch must be certified wholly inside or outside the lattice box; got a crossing or unresolved patch."
                )
            else:
                composition = _prepare_composition(
                    value, patch, domain, coordinate_budget
                )
                compositions.append(composition)
                patches.append(composition.template)
    except CoordinateEnclosureResourceError:
        raise
    except ValueError as error:
        return BRepDeformationResult(
            status="unsupported",
            certificate=certificate,
            parameter_map=None,
            reason=str(error),
            parameter_range=unknown_parameter_range("displacement_scale", str(error)),
        )
    parameter_map = BRepDeformationMap(
        deformation=value,
        compositions=tuple(compositions),
        patches=tuple(patches),
        model_id=model.model_id,
        feature_id=feature_id,
    )
    return BRepDeformationResult(
        status="certified",
        certificate=certificate,
        parameter_map=parameter_map,
        reason=certificate.reason,
        parameter_range=parameter_range,
    )


def _scaled_determinant(
    coordinates: tuple[polynomial.Polynomial, ...],
    original: tuple[polynomial.Polynomial, ...],
    candidate: Fraction,
    budget: CoordinateEnclosureBudget,
    /,
) -> polynomial.Polynomial:
    variables = polynomial.axes(4, budget=budget)
    scaled = []
    for source, target in zip(original, coordinates, strict=True):
        lifted_source: polynomial.Polynomial = {
            (*alpha, 0): coefficient for alpha, coefficient in source.items()
        }
        lifted_target: polynomial.Polynomial = {
            (*alpha, 0): coefficient for alpha, coefficient in target.items()
        }
        displacement = polynomial.add(
            lifted_target,
            polynomial.scale(lifted_source, -1, budget=budget),
            budget=budget,
        )
        movement = polynomial.multiply(
            displacement,
            polynomial.scale(variables[3], candidate, budget=budget),
            budget=budget,
        )
        scaled.append(polynomial.add(lifted_source, movement, budget=budget))
    jacobian = tuple(
        tuple(polynomial.derivative(item, axis, budget=budget) for axis in range(3))
        for item in scaled
    )
    # The identity reference is diagonal affine. Remove its positive volume
    # factor so cell ranges report physical det J, not parameter-volume density.
    numerator_bits, denominator_bits = polynomial.coefficient_profile(
        original, budget=budget
    )
    reserve_polynomial(3, 1, 3, 3 * (numerator_bits + denominator_bits), budget=budget)
    volume = original[0][(1, 0, 0)] * original[1][(0, 1, 0)] * original[2][(0, 0, 1)]
    determinant = polynomial.determinant(jacobian, budget=budget)
    return polynomial.scale(determinant, Fraction(1) / volume, budget=budget)


@checked
def deformation_feasibility_scale(
    value: PolynomialDeformation,
    /,
    *,
    maximum_scale: float,
    refinements: int,
    maximum_cells: int,
    coordinate_budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    """Largest dyadically certified inner scale after the declared refinements.

    Every accepted candidate certifies all spatial points AND every scale from
    zero to that candidate. The mathematical supremum can have det J = 0 and
    is not asserted feasible. Unresolved upper ends remain explicitly unknown.
    Physical determinant bounds and the tested bracket remain in ``evidence``.
    """
    positive_integer(refinements, "deformation_feasibility_scale.refinements")
    if not np.isfinite(maximum_scale) or maximum_scale <= 0:
        raise ValueError(
            "deformation_feasibility_scale.maximum_scale must be positive and finite; got an unsupported value."
        )
    if value.kind != "lattice":
        return unknown_parameter_range(
            "displacement_scale",
            "Named polynomial forms do not declare a lattice displacement scale.",
        )
    coordinates = _coordinates(value, coordinate_budget)
    bounds = np.asarray(value.box)
    variables = polynomial.axes(3, budget=coordinate_budget)
    original = tuple(
        polynomial.add(
            polynomial.constant(
                Fraction(float(bounds[0, axis])), 3, budget=coordinate_budget
            ),
            polynomial.scale(
                variable,
                Fraction(float(bounds[1, axis])) - Fraction(float(bounds[0, axis])),
                budget=coordinate_budget,
            ),
            budget=coordinate_budget,
        )
        for axis, variable in enumerate(variables)
    )
    lower, upper = Fraction(0), Fraction(maximum_scale)
    last_cells: tuple[BRepDeformationCell, ...] = ()
    feasible_cells = _certify_polynomial(
        polynomial.constant(1, 4, budget=coordinate_budget), 4, 1, coordinate_budget
    )
    for iteration in range(refinements + 1):
        candidate = upper if iteration == 0 else (lower + upper) / 2
        determinant = _scaled_determinant(
            coordinates, original, candidate, coordinate_budget
        )
        cells = _certify_polynomial(determinant, 4, maximum_cells, coordinate_budget)
        if all(cell.status == "certified" for cell in cells):
            lower, last_cells = candidate, cells
            feasible_cells = cells
            if iteration == 0:
                return _scale_parameter_range(
                    lower, upper, maximum_scale, last_cells, feasible_cells
                )
        else:
            upper, last_cells = candidate, cells
    return _scale_parameter_range(lower, upper, maximum_scale, last_cells, feasible_cells)
