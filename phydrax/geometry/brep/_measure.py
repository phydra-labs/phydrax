#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified moments and Green-reduced trimmed-face quadrature estimates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import assert_never, final, Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.jet import jet

from ..._jax_config import x64_enabled
from ..._numerics._quadrature_rules import gauss_legendre_data
from ..._strict import StrictModule
from ..._validation import positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from ...units._dimension import LENGTH, MASS
from ...units._unit import UnitDefinition
from .._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import fraction_down, fraction_up, rational_dot, rational_vector
from ._affine_query import _plane
from ._exact import rational_determinant, square_root_interval
from ._intersection_curve import (
    affine_curve_coefficients,
    AffinePCurve,
    curve_pieces_for_interval,
    IntersectionCurve,
    IntersectionPCurve,
)
from ._interval_derivatives import prepare_taylor_derivative_interval
from ._model import BRepEntityId, BRepGeometry, BRepModel, BRepPCurve
from ._patches import (
    _array_interval_bounds,
    _bernstein,
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinCurvePiece,
    BernsteinSurfacePiece,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    coefficient_enclosures,
    curve_value_and_derivative_host,
    EllipseCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    interval_cross,
    interval_dot,
    interval_norm,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    RationalBezierPiece,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    surface_value_and_differential_host,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ._placed import PlacedCurve, PlacedSurface
from ._query_membership import full_rectangle
from ._query_types import BRepQueryBudget


_MomentSet: TypeAlias = Literal["volume", "mass_properties"]


def _moment_shapes(moment_set: _MomentSet, /) -> tuple[tuple[int, ...], ...]:
    match moment_set:
        case "volume":
            return (), (3,)
        case "mass_properties":
            return (), (3,), (3, 3), (), (3,), (3,)
        case _:
            assert_never(moment_set)


def _scalar_interval(value: float, /) -> tuple[np.float64, np.float64]:
    scalar = np.float64(value)
    return scalar, scalar


def _orthonormal_round_axes(
    patch: SpherePatch | TorusPatch, /
) -> tuple[tuple[np.ndarray, ...], float] | None:
    axes = tuple(
        np.asarray(axis) for axis in (patch.first_axis, patch.second_axis, patch.axis)
    )
    exact_axes = tuple(rational_vector(axis) for axis in axes)
    if any(
        rational_dot(a, b) != Fraction(i == j)
        for i, a in enumerate(exact_axes)
        for j, b in enumerate(exact_axes)
    ):
        return None
    return axes, float(rational_determinant(exact_axes[0], exact_axes[1], exact_axes[2]))


def _closed_round_face(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    center: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...] | None:
    """Sphere/torus moments after rational frame and chart-coverage proofs."""
    patch = model.patches[face]
    match patch:
        case SpherePatch() | TorusPatch():
            pass
        case _:
            return None
    budget.reserve(4096, 32768)
    box = np.asarray(model.parameter_bounds)[face]
    if not full_rectangle(geometry, face, patch, box):
        return None
    if box[0, 0] != 0 or box[1, 0] != 2 * np.pi:
        return None
    match patch:
        case SpherePatch():
            if box[0, 1] != -np.pi / 2 or box[1, 1] != np.pi / 2:
                return None
        case TorusPatch():
            if box[0, 1] != 0 or box[1, 1] != 2 * np.pi:
                return None
    frame = _orthonormal_round_axes(patch)
    if frame is None:
        return None
    axes, handedness = frame
    pi = (
        np.nextafter(np.float64(np.pi), -np.inf),
        np.nextafter(np.float64(np.pi), np.inf),
    )
    orientation = _scalar_interval(float(model.orientation[face]) * handedness)
    match patch:
        case SpherePatch():
            r = _scalar_interval(float(patch.radius))
            r2 = interval_multiply(r, r)
            area = interval_multiply(interval_multiply(_scalar_interval(4), pi), r2)
            volume = interval_divide(interval_multiply(area, r), _scalar_interval(3))
            if moment_set == "volume":
                signed = _array_interval_bounds(interval_multiply(volume, orientation))
                return signed, (np.full(3, signed[0]), np.full(3, signed[1]))
            radial = axial = interval_divide(r2, _scalar_interval(5))
        case TorusPatch():
            r = _scalar_interval(float(patch.minor_radius))
            major = _scalar_interval(float(patch.major_radius))
            r2 = interval_multiply(r, r)
            factor = interval_multiply(interval_multiply(pi, pi), major)
            volume = interval_multiply(interval_multiply(_scalar_interval(2), factor), r2)
            if moment_set == "volume":
                signed = _array_interval_bounds(interval_multiply(volume, orientation))
                return signed, (np.full(3, signed[0]), np.full(3, signed[1]))
            area = interval_multiply(interval_multiply(_scalar_interval(4), factor), r)
            radial = interval_add(
                interval_divide(interval_multiply(major, major), _scalar_interval(2)),
                interval_divide(
                    interval_multiply(_scalar_interval(3), r2), _scalar_interval(8)
                ),
            )
            axial = interval_divide(r2, _scalar_interval(4))
    delta = _array_interval_bounds(
        interval_subtract(
            (np.asarray(patch.center), np.asarray(patch.center)), (center, center)
        )
    )
    covariance = (np.zeros((3, 3)), np.zeros((3, 3)))
    for axis, variance in zip(axes, (radial, radial, axial), strict=True):
        outer = interval_multiply(
            (axis[:, None], axis[:, None]), (axis[None, :], axis[None, :])
        )
        covariance = _array_interval_bounds(
            interval_add(covariance, interval_multiply(outer, variance))
        )
    second = interval_multiply(
        volume,
        interval_add(
            covariance,
            interval_multiply(
                (delta[0][:, None], delta[1][:, None]),
                (delta[0][None, :], delta[1][None, :]),
            ),
        ),
    )
    oriented_volume = interval_multiply(volume, orientation)
    first = interval_multiply(oriented_volume, delta)
    second = interval_multiply(second, orientation)
    return tuple(
        _array_interval_bounds(value)
        for value in (
            oriented_volume,
            first,
            second,
            area,
            interval_multiply(area, (np.asarray(patch.center), np.asarray(patch.center))),
            (
                np.full(3, oriented_volume[0]),
                np.full(3, oriented_volume[1]),
            ),
        )
    )


@final
class BRepMassProperties(StrictModule):
    """Certified geometric moments in source coordinates, in model solid order.

    Inertia is geometric inertia about the enclosed centroid, not mass-weighted
    inertia. Every enclosure has its lower/upper axis immediately after the
    entity axis. No positive-volume claim is made from quadrature differences.
    """

    model_id: str = eqx.field(static=True)
    volume: Array
    volume_enclosures: Array
    first_moment_enclosures: Array
    second_moment_enclosures: Array
    centroid: Array
    centroid_enclosures: Array
    inertia: Array
    inertia_enclosures: Array
    face_area_enclosures: Array
    face_centroid_enclosures: Array
    mass: Array | None
    mass_enclosures: Array | None
    mass_inertia: Array | None
    mass_inertia_enclosures: Array | None
    density_unit: UnitDefinition | None
    coordinate_unit: UnitDefinition

    @property
    def volume_error_bound(self) -> Array:
        return jnp.nextafter(
            jnp.max(jnp.abs(self.volume_enclosures - self.volume[:, None]), axis=1),
            jnp.inf,
        )

    @property
    def centroid_error_bound(self) -> Array:
        return jnp.nextafter(
            jnp.max(
                jnp.abs(self.centroid_enclosures - self.centroid[:, None, :]), axis=1
            ),
            jnp.inf,
        )

    @property
    def inertia_error_bound(self) -> Array:
        return jnp.nextafter(
            jnp.max(
                jnp.abs(self.inertia_enclosures - self.inertia[:, None, :, :]), axis=1
            ),
            jnp.inf,
        )


def _mass_integrands(
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    center: np.ndarray,
    orientation: float,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    points = patch.bounding_box(box, budget=budget)
    x = _array_interval_bounds(
        interval_subtract((points[0], points[1]), (center, center))
    )
    lower, upper = patch.derivative_bounds(box, budget=budget)
    raw_normal = interval_cross((lower[:, 0], upper[:, 0]), (lower[:, 1], upper[:, 1]))
    normal = _array_interval_bounds(
        interval_multiply(raw_normal, (np.float64(orientation), np.float64(orientation)))
    )
    flux = interval_dot(x, normal)
    axis_flux = _array_interval_bounds(interval_multiply(x, normal))
    if moment_set == "volume":
        return (
            _array_interval_bounds(interval_divide(flux, (np.float64(3), np.float64(3)))),
            axis_flux,
        )
    area = interval_norm(raw_normal)
    first = _array_interval_bounds(
        interval_divide(
            _array_interval_bounds(interval_multiply(x, flux)),
            (np.float64(4), np.float64(4)),
        )
    )
    outer = _array_interval_bounds(
        interval_multiply((x[0][:, None], x[1][:, None]), (x[0][None, :], x[1][None, :]))
    )
    second = _array_interval_bounds(
        interval_divide(
            _array_interval_bounds(interval_multiply(outer, flux)),
            (np.float64(5), np.float64(5)),
        )
    )
    return (
        _array_interval_bounds(interval_divide(flux, (np.float64(3), np.float64(3)))),
        first,
        second,
        area,
        _array_interval_bounds(interval_multiply((points[0], points[1]), area)),
        axis_flux,
    )


_Polynomial: TypeAlias = dict[tuple[int, int], Fraction]


def _polynomial_product(first: _Polynomial, second: _Polynomial, /) -> _Polynomial:
    result: _Polynomial = {}
    for (u, v), a in first.items():
        for (s, t), b in second.items():
            key = u + s, v + t
            result[key] = result.get(key, Fraction(0)) + a * b
    return result


def _affine_nurbs_coefficients(
    patch: BSplineSurfacePatch, /
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Prove affine reproduction from rational Greville coordinates and controls."""
    weights = np.asarray(patch.weights)
    if not np.all(weights == weights.flat[0]):
        return None
    coordinates = []
    for knots, degree in (
        (patch.u_knots, patch.u_degree),
        (patch.v_knots, patch.v_degree),
    ):
        source = rational_vector(np.asarray(knots))
        coordinates.append(
            tuple(
                sum(source[i + 1 : i + degree + 1], Fraction(0)) / degree
                for i in range(len(source) - degree - 1)
            )
        )
    u, v = coordinates
    if u[0] == u[-1] or v[0] == v[-1]:
        return None
    controls = np.asarray(patch.control_points)
    corner = rational_vector(controls[0, 0])
    first = tuple(
        (a - b) / (u[-1] - u[0])
        for a, b in zip(rational_vector(controls[-1, 0]), corner, strict=True)
    )
    second = tuple(
        (a - b) / (v[-1] - v[0])
        for a, b in zip(rational_vector(controls[0, -1]), corner, strict=True)
    )
    origin = tuple(
        a - u[0] * b - v[0] * c for a, b, c in zip(corner, first, second, strict=True)
    )
    for i, s in enumerate(u):
        for j, t in enumerate(v):
            expected = tuple(
                a + s * b + t * c for a, b, c in zip(origin, first, second, strict=True)
            )
            if rational_vector(controls[i, j]) != expected:
                return None
    return origin, first, second


def _line_monomial_integral(
    u: tuple[Fraction, Fraction],
    v: tuple[Fraction, Fraction],
    first: Fraction,
    last: Fraction,
    power: tuple[int, int],
    /,
) -> Fraction:
    """Rational Green antiderivative on one affine boundary parameter span."""
    a, b = power[0] + 1, power[1]
    total = Fraction(0)
    for i in range(a + 1):
        for j in range(b + 1):
            coefficient = (
                comb(a, i)
                * comb(b, j)
                * u[0] ** (a - i)
                * u[1] ** i
                * v[0] ** (b - j)
                * v[1] ** j
            )
            exponent = i + j + 1
            total += coefficient * (last**exponent - first**exponent) / exponent
    return total * v[1] / a


def _affine_trim_monomials(
    geometry: BRepGeometry,
    coedges: tuple[int, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> dict[tuple[int, int], Fraction] | None:
    """Integrate the degree-three moment basis without a sampled trim mask."""
    spans = []
    for coedge in coedges:
        curve = geometry.pcurves[coedge]
        affine = (
            affine_curve_coefficients(curve, np.zeros(2, dtype=np.float64))
            if isinstance(curve, AbstractCurve)
            else None
        )
        endpoints = geometry.edge_parameter_enclosure(
            geometry.coedge_edges[coedge], budget=budget
        )
        if affine is None or np.any(endpoints[:, 0] != endpoints[:, 1]):
            return None
        spans.append(
            (affine, rational_vector(endpoints[:, 0]), geometry.coedge_senses[coedge])
        )
    integrals: dict[tuple[int, int], Fraction] = {}
    for a in range(4):
        for b in range(4 - a):
            integrals[a, b] = sum(
                (
                    sense
                    * _line_monomial_integral(
                        (origin[0], direction[0]),
                        (origin[1], direction[1]),
                        endpoints[0],
                        endpoints[1],
                        (a, b),
                    )
                    for (origin, direction), endpoints, sense in spans
                ),
                Fraction(0),
            )
    return integrals


def _affine_flux_polynomials(
    coefficients: tuple[tuple[Fraction, ...], ...],
    center: np.ndarray,
    orientation: float,
    /,
) -> tuple[
    tuple[Fraction, ...], tuple[Fraction, ...], tuple[_Polynomial, ...], _Polynomial
]:
    origin, du, dv = coefficients
    raw_normal = tuple(
        du[(i + 1) % 3] * dv[(i + 2) % 3] - du[(i + 2) % 3] * dv[(i + 1) % 3]
        for i in range(3)
    )
    normal = tuple(Fraction(orientation) * value for value in raw_normal)
    x = tuple(
        {(0, 0): a - c, (1, 0): b, (0, 1): d}
        for a, b, d, c in zip(origin, du, dv, rational_vector(center), strict=True)
    )
    flux = {
        key: sum((normal[i] * x[i][key] for i in range(3)), Fraction(0))
        for key in ((0, 0), (1, 0), (0, 1))
    }
    return raw_normal, normal, x, flux


def _affine_face_moments(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    center: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...] | None:
    """Polynomial inner antiderivatives and rational affine trim integrals."""
    patch = model.patches[face]
    coedges = tuple(coedge for loop in geometry.face_loops[face] for coedge in loop)
    source_size = (
        patch.control_points.size if isinstance(patch, BSplineSurfacePatch) else 12
    )
    budget.reserve(32 * (source_size + len(coedges)), 256 * (source_size + len(coedges)))
    coefficients = (
        _affine_nurbs_coefficients(patch)
        if isinstance(patch, BSplineSurfacePatch)
        else _plane(patch)
    )
    if coefficients is None:
        return None
    source_integrals = _affine_trim_monomials(geometry, coedges, budget)
    if source_integrals is None:
        return None
    integrals: dict[tuple[int, int], Fraction] = source_integrals
    raw_normal, normal, x, flux = _affine_flux_polynomials(
        coefficients, center, float(model.orientation[face])
    )

    def integral(polynomial: _Polynomial) -> Fraction:
        return sum(
            (value * integrals[key] for key, value in polynomial.items()), Fraction(0)
        )

    def bounds(
        values: tuple[Fraction, ...], shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(
                [fraction_down(value) for value in values], dtype=np.float64
            ).reshape(shape),
            np.asarray(
                [fraction_up(value) for value in values], dtype=np.float64
            ).reshape(shape),
        )

    volume = bounds((integral(flux) / 3,), ())
    axes = bounds(tuple(normal[i] * integral(x[i]) for i in range(3)), (3,))
    if moment_set == "volume":
        return volume, axes
    first = bounds(
        tuple(integral(_polynomial_product(value, flux)) / 4 for value in x), (3,)
    )
    second = bounds(
        tuple(
            integral(_polynomial_product(_polynomial_product(a, b), flux)) / 5
            for a in x
            for b in x
        ),
        (3, 3),
    )
    lower_norm, upper_norm = square_root_interval(rational_dot(raw_normal, raw_normal))
    norm = np.float64(lower_norm), np.float64(upper_norm)
    domain_area = bounds((integrals[0, 0],), ())
    area = _array_interval_bounds(interval_multiply(norm, domain_area))
    position_integrals = bounds(
        tuple(
            integral(value) + c * integrals[0, 0]
            for value, c in zip(x, rational_vector(center), strict=True)
        ),
        (3,),
    )
    barycenter = _array_interval_bounds(interval_multiply(norm, position_integrals))
    return volume, first, second, area, barycenter, axes


@final
class _MomentDensity(StrictModule):
    patch: BernsteinSurfacePiece
    center: Array
    orientation: Array
    denominator_floor: Array
    moment_set: _MomentSet = eqx.field(static=True)

    def __init__(
        self,
        patch: BernsteinSurfacePiece,
        center: Array,
        orientation: Array,
        moment_set: _MomentSet,
    ) -> None:
        minimum_weight = np.min(np.asarray(patch.controls_lower)[..., 3])
        if minimum_weight <= 0:
            raise ValueError(
                "BRepMassProperties.denominator must have positive Bernstein weight bounds; "
                f"got lower weight {minimum_weight!r}."
            )
        floor = max(
            np.float64(0),
            np.nextafter(np.float64(minimum_weight / 2), -np.inf),
        )
        self.patch = patch
        self.center = center
        self.orientation = orientation
        self.denominator_floor = jax.device_put(floor)
        self.moment_set = parse(moment_set, _MomentSet, "BRepMassProperties.moment_set")

    def _homogeneous_jet(self, parameters: Array, /) -> tuple[Array, Array, Array]:
        width = self.patch.upper - self.patch.lower
        local = (parameters - self.patch.lower) / width
        controls = self.patch.controls
        u_degree, v_degree = controls.shape[0] - 1, controls.shape[1] - 1
        first = _bernstein(local[0], u_degree)
        second = _bernstein(local[1], v_degree)
        rows = first @ controls.reshape((controls.shape[0], -1))
        value = second @ rows.reshape((controls.shape[1], 4))
        if u_degree:
            difference = controls[1:] - controls[:-1]
            rows = _bernstein(local[0], u_degree - 1) @ difference.reshape((u_degree, -1))
            du = (second @ rows.reshape((controls.shape[1], 4))) * u_degree / width[0]
        else:
            du = jnp.zeros_like(value)
        if v_degree:
            difference = controls[:, 1:] - controls[:, :-1]
            rows = first @ difference.reshape((controls.shape[0], -1))
            dv = (
                (_bernstein(local[1], v_degree - 1) @ rows.reshape((v_degree, 4)))
                * v_degree
                / width[1]
            )
        else:
            dv = jnp.zeros_like(value)
        return value, du, dv

    def evaluate(self, parameters: Array, /) -> Array:
        homogeneous, first, second = self._homogeneous_jet(parameters)
        # Bernstein weights are nonnegative and sum to one on the admitted
        # chart. W >= minimum_weight > floor makes this constraint inactive
        # for the source while retaining positivity in interval divisions.
        denominator = self.denominator_floor + jnp.abs(
            homogeneous[3] - self.denominator_floor
        )
        point = homogeneous[:3] / denominator
        du = (first[:3] - point * first[3]) / denominator
        dv = (second[:3] - point * second[3]) / denominator
        x = point - self.center
        normal = jnp.cross(du, dv) * self.orientation
        flux = x @ normal
        axes = x * normal
        if self.moment_set == "volume":
            return jnp.concatenate((jnp.reshape(flux / 3, (1,)), axes))
        return jnp.concatenate(
            (
                jnp.reshape(flux / 3, (1,)),
                x * flux / 4,
                jnp.reshape(x[:, None] * x[None, :] * flux / 5, (9,)),
                axes,
            )
        )


def _trim_point_and_derivative(
    pcurve: AbstractCurve, parameter: Array, /
) -> tuple[Array, Array]:
    """Differentiate homogeneous span controls before the rational quotient."""
    if isinstance(pcurve, AffinePCurve) and isinstance(pcurve.curve, AbstractCurve):
        point, derivative = _trim_point_and_derivative(pcurve.curve, parameter)
        return (
            pcurve.matrix_value @ point + pcurve.offset_value,
            pcurve.matrix_value @ derivative,
        )
    if isinstance(pcurve, BernsteinCurvePiece):
        degree = pcurve.controls.shape[0] - 1
        local = (parameter - pcurve.lower) / (pcurve.upper - pcurve.lower)
        homogeneous = _bernstein(local, degree) @ pcurve.controls
        floor = jnp.min(pcurve.controls_lower[:, -1]) / 2
        denominator = floor + jnp.abs(homogeneous[-1] - floor)
        point = homogeneous[:-1] / denominator
        if degree:
            difference = pcurve.controls[1:] - pcurve.controls[:-1]
            derivative = (
                (_bernstein(local, degree - 1) @ difference)
                * degree
                / (pcurve.upper - pcurve.lower)
            )
        else:
            derivative = jnp.zeros_like(homogeneous)
        velocity = (derivative[:-1] - point * derivative[-1]) / denominator
        return point, velocity
    return jax.jvp(pcurve.evaluate, (parameter,), (jnp.ones_like(parameter),))


@final
class _GreenMomentDensity(StrictModule):
    density: _MomentDensity
    pcurve: AbstractCurve
    lower_u: Array
    upper_u: Array
    sense: Array
    final_span: bool = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)

    def evaluate(self, parameters: Array, /) -> Array:
        q, t = parameters[0], parameters[1]
        boundary, derivative = _trim_point_and_derivative(self.pcurve, t)
        endpoint = boundary[0] if self.final_span else self.upper_u
        width = endpoint - self.lower_u
        velocity = derivative[1] * self.sense
        if self.polynomial_degree < 0:
            uv = jnp.stack((self.lower_u + q * width, boundary[1]))
            return self.density.evaluate(uv) * width * velocity
        uv = jnp.stack((self.lower_u, boundary[1]))
        tangent = jnp.asarray((1.0, 0.0), dtype=jnp.float64)
        value, coefficients = jet(
            self.density.evaluate,
            (uv,),
            ((tangent,) + (jnp.zeros_like(uv),) * (self.polynomial_degree - 1),),
            factorial_scaled=False,
        )
        primitive = value * width
        for order, coefficient in enumerate(coefficients, start=1):
            primitive = primitive + coefficient * width ** (order + 1) / (order + 1)
        return primitive * velocity


@dataclass(frozen=True, slots=True)
class _GaussGreenRule:
    value: PreparedIntervalFunction
    outer_derivative: PreparedIntervalFunction
    inner_derivative: PreparedIntervalFunction | None
    size: int

    def _remainder_bounds(
        self,
        box: np.ndarray,
        width: tuple[np.ndarray, np.ndarray],
        /,
    ) -> np.ndarray:
        """The two-point rule has remainder length**5 * sup|f''''| / 4320."""
        rational = self.inner_derivative is not None
        errors = np.zeros((2, self.size), dtype=np.float64)
        for axis, derivative in ((1, self.outer_derivative), (0, self.inner_derivative)):
            if derivative is None:
                continue
            bounds = derivative.evaluate(box[0][None], box[1][None])
            magnitude = np.maximum(np.abs(bounds[0][0]), np.abs(bounds[1][0]))
            power = _scalar_interval(1)
            for _ in range(5):
                power = _array_interval_bounds(
                    interval_multiply(power, (width[0][axis], width[1][axis]))
                )
            factor = interval_divide(power, _scalar_interval(4320))
            other = (
                (width[0][1 - axis], width[1][1 - axis])
                if rational
                else _scalar_interval(1)
            )
            remainder = interval_multiply(
                interval_multiply(factor, other), (magnitude, magnitude)
            )
            errors[axis] = np.nextafter(remainder[1], np.inf)
        return errors

    def evaluate(
        self, box: np.ndarray, budget: CoordinateEnclosureBudget, /
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Outer boundary Gauss plus polynomial or certified inner antiderivative."""
        budget.reserve(8192, 1_048_576)
        rational = self.inner_derivative is not None
        root = square_root_interval(Fraction(1, 3))
        lower_nodes, upper_nodes = (-root[1], root[0]), (-root[0], root[1])
        result = (
            np.zeros(self.size, dtype=np.float64),
            np.zeros(self.size, dtype=np.float64),
        )
        width = _array_interval_bounds(
            interval_subtract((box[1], box[1]), (box[0], box[0]))
        )
        half = _array_interval_bounds(interval_divide(width, _scalar_interval(2)))
        midpoint = _array_interval_bounds(interval_add((box[0], box[0]), half))
        for t in range(2):
            for q in range(2 if rational else 1):
                node = (
                    np.asarray(
                        (lower_nodes[q] if rational else 0, lower_nodes[t]),
                        dtype=np.float64,
                    ),
                    np.asarray(
                        (upper_nodes[q] if rational else 0, upper_nodes[t]),
                        dtype=np.float64,
                    ),
                )
                parameter = _array_interval_bounds(
                    interval_add(midpoint, interval_multiply(half, node))
                )
                bounds = self.value.evaluate(parameter[0][None], parameter[1][None])
                weight = interval_multiply(
                    (half[0][1], half[1][1]),
                    (half[0][0], half[1][0]) if rational else _scalar_interval(1),
                )
                result = _array_interval_bounds(
                    interval_add(
                        result, interval_multiply((bounds[0][0], bounds[1][0]), weight)
                    )
                )
        errors = self._remainder_bounds(box, width)
        error = np.nextafter(errors[0] + errors[1], np.inf)
        return _array_interval_bounds(interval_add(result, (-error, error))), errors


def _prepare_gauss_green_rule(
    evaluator: _GreenMomentDensity,
    budget: CoordinateEnclosureBudget,
    /,
) -> _GaussGreenRule:
    coefficients = coefficient_enclosures(evaluator)
    budget.reserve(8192, 1_048_576)
    return _GaussGreenRule(
        prepare_interval_function(
            evaluator.evaluate, 2, batch_capacity=1, constant_bounds=coefficients
        ),
        prepare_taylor_derivative_interval(
            evaluator, 2, 1, 4, batch_capacity=1, constant_bounds=coefficients
        ),
        prepare_taylor_derivative_interval(
            evaluator, 2, 0, 4, batch_capacity=1, constant_bounds=coefficients
        )
        if evaluator.polynomial_degree < 0
        else None,
        4 if evaluator.density.moment_set == "volume" else 16,
    )


def _refine_gauss_boundary(
    rule: _GaussGreenRule,
    first: float,
    last: float,
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Spend the declared piece allowance on the largest certified remainder."""
    box = np.asarray(((0, first), (1, last)), dtype=np.float64)
    value, errors = rule.evaluate(box, budget)
    cells = [(box, value, errors)]
    while len(cells) < subdivisions:
        index = max(range(len(cells)), key=lambda i: float(np.max(cells[i][2])))
        box, _, errors = cells[index]
        axis = int(np.argmax(np.max(errors, axis=1)))
        midpoint = box[0, axis] / 2 + box[1, axis] / 2
        if not box[0, axis] < midpoint < box[1, axis]:
            break
        first_box, second_box = box.copy(), box.copy()
        first_box[1, axis], second_box[0, axis] = midpoint, midpoint
        first_value, first_errors = rule.evaluate(first_box, budget)
        second_value, second_errors = rule.evaluate(second_box, budget)
        cells[index : index + 1] = [
            (first_box, first_value, first_errors),
            (second_box, second_value, second_errors),
        ]
    total = np.zeros(rule.size, dtype=np.float64), np.zeros(rule.size, dtype=np.float64)
    for _, value, _ in cells:
        total = _array_interval_bounds(interval_add(total, value))
    return total


def _green_interval_piece(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    coedge: int,
    center: np.ndarray,
    first: float,
    last: float,
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Certified residual strip at a knot crossing, including rooted trims."""
    budget.reserve(1024 * subdivisions, 4096 * subdivisions)
    u0 = float(model.parameter_bounds[face, 0, 0])
    pcurve = geometry.pcurves[coedge]
    uv = pcurve.bounding_box(first, last, budget=budget)
    derivative = pcurve.derivative_bounds(first, last, budget=budget)
    du = interval_subtract((uv[0, 0], uv[1, 0]), _scalar_interval(u0))
    jacobian = interval_multiply(
        interval_multiply(du, (derivative[0][1], derivative[1][1])),
        _scalar_interval(geometry.coedge_senses[coedge]),
    )
    span = interval_subtract(_scalar_interval(last), _scalar_interval(first))
    weight = interval_divide(
        interval_multiply(jacobian, span), _scalar_interval(subdivisions)
    )
    totals = tuple(
        (np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64))
        for shape in _moment_shapes(moment_set)
    )
    for q in range(subdivisions):
        fraction = (
            max(np.float64(0), np.nextafter(np.float64(q / subdivisions), -np.inf)),
            min(np.float64(1), np.nextafter(np.float64((q + 1) / subdivisions), np.inf)),
        )
        u = interval_add(_scalar_interval(u0), interval_multiply(du, fraction))
        box = np.asarray(((u[0], uv[0, 1]), (u[1], uv[1, 1])), dtype=np.float64)
        values = _mass_integrands(
            model.patches[face],
            box,
            center,
            float(model.orientation[face]),
            budget,
            moment_set=moment_set,
        )
        totals = tuple(
            _array_interval_bounds(interval_add(old, interval_multiply(value, weight)))
            for old, value in zip(totals, values, strict=True)
        )
    return totals


def _refine_knot_crossing_cuts(
    patch: BSplineSurfacePatch,
    pcurve: AbstractCurve,
    cuts: set[float],
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[float, ...]:
    """Isolate curved knot crossings into certified residual strips."""
    maximum_pieces = max(len(cuts) - 1, subdivisions**2)
    budget.reserve(64 * maximum_pieces, 512 * maximum_pieces)
    walls = tuple(
        np.unique(np.asarray(knots))[1:-1] for knots in (patch.u_knots, patch.v_knots)
    )
    ordered = sorted(cuts)
    pending = list(zip(ordered[:-1], ordered[1:], strict=True))
    accepted: list[tuple[float, float]] = []
    while pending:
        first, last = pending.pop()
        box = pcurve.bounding_box(first, last, budget=budget)
        crossed = any(
            np.any((box[0, axis] < knots) & (knots < box[1, axis]))
            for axis, knots in enumerate(walls)
        )
        midpoint = first / 2 + last / 2
        if (
            crossed
            and len(pending) + len(accepted) + 1 < maximum_pieces
            and first < midpoint < last
        ):
            pending.extend(((midpoint, last), (first, midpoint)))
        else:
            accepted.append((first, last))
    return tuple(sorted({value for interval in accepted for value in interval}))


_TAYLOR_PCURVES = (
    LineCurve,
    CircleCurve,
    EllipseCurve,
    ParabolaCurve,
    HyperbolaCurve,
    BernsteinCurvePiece,
)


def _has_taylor_trim_source(pcurve: BRepPCurve, /) -> bool:
    if not isinstance(pcurve, AbstractCurve):
        return False
    if isinstance(pcurve, AffinePCurve):
        return isinstance(pcurve.curve, AbstractCurve) and _has_taylor_trim_source(
            pcurve.curve
        )
    return isinstance(pcurve, (BSplineCurve, *_TAYLOR_PCURVES))


def _trim_native_knots(
    pcurve: AbstractCurve, budget: CoordinateEnclosureBudget, /
) -> np.ndarray:
    if isinstance(pcurve, AffinePCurve) and isinstance(pcurve.curve, AbstractCurve):
        return _trim_native_knots(pcurve.curve, budget)
    if isinstance(pcurve, BSplineCurve):
        budget.reserve(pcurve.knots.size, 16 * pcurve.knots.size)
        return np.asarray(pcurve.knots, dtype=np.float64)
    return np.empty(0, dtype=np.float64)


def _taylor_trim_source(
    pcurve: AbstractCurve,
    first: float,
    last: float,
    budget: CoordinateEnclosureBudget,
    /,
) -> AbstractCurve | None:
    """Carry validated spline-span controls rather than differentiating a guard."""
    if not _has_taylor_trim_source(pcurve):
        return None
    pieces = curve_pieces_for_interval(pcurve, first, last, coordinate_budget=budget)
    return pieces[0].evaluator if len(pieces) == 1 else None


def _nurbs_boundary_cuts(
    patch: BSplineSurfacePatch,
    pcurve: AbstractCurve,
    first: float,
    last: float,
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[float, ...]:
    """Split source knots and enclose nonbinary affine knot-line crossings."""
    budget.reserve(patch.u_knots.size + patch.v_knots.size + 16, 8192)
    cuts = {first, last}
    cuts.update(
        float(knot) for knot in _trim_native_knots(pcurve, budget) if first < knot < last
    )
    affine = affine_curve_coefficients(pcurve, np.zeros(2, dtype=np.float64))
    if affine is not None:
        for axis, knots in enumerate((patch.u_knots, patch.v_knots)):
            if affine[1][axis] == 0:
                continue
            for knot in rational_vector(np.asarray(knots)):
                root = (knot - affine[0][axis]) / affine[1][axis]
                if Fraction(first) < root < Fraction(last):
                    cuts.update((fraction_down(root), fraction_up(root)))
    else:
        return _refine_knot_crossing_cuts(patch, pcurve, cuts, subdivisions, budget)
    return tuple(sorted(cuts))


def _boundary_parameter_box(
    pcurve: AbstractCurve,
    first: float,
    last: float,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    affine = affine_curve_coefficients(pcurve, np.zeros(2, dtype=np.float64))
    if affine is None:
        return pcurve.bounding_box(first, last, budget=budget)
    exact = tuple(
        tuple(origin + t * direction for t in (Fraction(first), Fraction(last)))
        for origin, direction in zip(*affine, strict=True)
    )
    return np.asarray(
        [
            [fraction_down(min(axis)) for axis in exact],
            [fraction_up(max(axis)) for axis in exact],
        ],
        dtype=np.float64,
    )


def _unpack_green_moments(
    value: tuple[np.ndarray, np.ndarray],
    residual: tuple[tuple[np.ndarray, np.ndarray], ...],
    moment_set: _MomentSet,
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    shapes = ((), (3,)) if moment_set == "volume" else ((), (3,), (3, 3), (3,))
    positions = (0, 1) if moment_set == "volume" else (0, 1, 2, 5)
    result = list(residual)
    offset = 0
    for position, shape in zip(positions, shapes, strict=True):
        size = 1 if not shape else int(np.prod(shape))
        result[position] = (
            value[0][offset : offset + size].reshape(shape),
            value[1][offset : offset + size].reshape(shape),
        )
        offset += size
    return tuple(result)


def _smooth_green_sources(
    pieces: tuple[RationalBezierPiece, ...],
    box: np.ndarray,
    lower_u: float,
    /,
) -> tuple[tuple[int, RationalBezierPiece, bool], ...]:
    """A boundary piece must lie in one chart before its derivative is bounded."""
    sources = tuple(
        (index, piece)
        for index, piece in enumerate(pieces)
        if piece.parameter_bounds[1][0] <= box[0, 1]
        and box[1, 1] <= piece.parameter_bounds[1][1]
        and piece.parameter_bounds[0][0] < box[1, 0]
        and piece.parameter_bounds[0][1] > lower_u
    )
    final = tuple(
        (index, piece)
        for index, piece in sources
        if piece.parameter_bounds[0][0]
        <= box[0, 0]
        <= box[1, 0]
        <= piece.parameter_bounds[0][1]
    )
    if not final:
        return ()
    selected_index, selected = final[0]
    return tuple(
        (index, piece, index == selected_index)
        for index, piece in sources
        if piece.parameter_bounds[0][0] <= selected.parameter_bounds[0][0]
    )


def _prepare_nurbs_span_rule(
    piece: RationalBezierPiece,
    pcurve: AbstractCurve,
    center: np.ndarray,
    orientation: float,
    sense: int,
    lower_u: float,
    final_span: bool,
    degree: int,
    moment_set: _MomentSet,
    budget: CoordinateEnclosureBudget,
    /,
) -> _GaussGreenRule:
    bounds = np.asarray(piece.parameter_bounds, dtype=np.float64).T
    source = BernsteinSurfacePiece(
        piece.homogeneous_controls,
        bounds[0],
        bounds[1],
        controls_lower=piece.homogeneous_lower,
        controls_upper=piece.homogeneous_upper,
    )
    evaluator = _GreenMomentDensity(
        _MomentDensity(
            source,
            jax.device_put(center),
            jax.device_put(np.float64(orientation)),
            moment_set,
        ),
        pcurve,
        jax.device_put(np.float64(max(lower_u, bounds[0, 0]))),
        jax.device_put(np.float64(bounds[1, 0])),
        jax.device_put(np.float64(sense)),
        final_span,
        degree,
    )
    return _prepare_gauss_green_rule(evaluator, budget)


def _integrate_nurbs_boundary_span(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    coedge: int,
    center: np.ndarray,
    pcurve: AbstractCurve,
    pieces: tuple[RationalBezierPiece, ...],
    rules: dict[tuple[int, int, bool, float, float], _GaussGreenRule],
    first: float,
    last: float,
    subdivisions: int,
    degree: int,
    budget: CoordinateEnclosureBudget,
    moment_set: _MomentSet,
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    u0 = float(model.parameter_bounds[face, 0, 0])
    uv = _boundary_parameter_box(pcurve, first, last, budget)
    sources = _smooth_green_sources(pieces, uv, u0)
    source_pcurve = _taylor_trim_source(pcurve, first, last, budget)
    residual = _green_interval_piece(
        model,
        geometry,
        face,
        coedge,
        center,
        first,
        last,
        subdivisions if not sources or source_pcurve is None else 1,
        budget,
        moment_set=moment_set,
    )
    if not sources or source_pcurve is None:
        return residual
    size = 4 if moment_set == "volume" else 16
    value = np.zeros(size, dtype=np.float64), np.zeros(size, dtype=np.float64)
    for index, piece, final_span in sources:
        key = coedge, index, final_span, first, last
        if key not in rules:
            rules[key] = _prepare_nurbs_span_rule(
                piece,
                source_pcurve,
                center,
                float(model.orientation[face]),
                geometry.coedge_senses[coedge],
                u0,
                final_span,
                degree,
                moment_set,
                budget,
            )
        enclosure = _refine_gauss_boundary(rules[key], first, last, subdivisions, budget)
        value = _array_interval_bounds(interval_add(value, enclosure))
    return _unpack_green_moments(value, residual, moment_set)


def _polynomial_moment_degree(
    patch: BSplineSurfacePatch, moment_set: _MomentSet, /
) -> int:
    """Constant weights prove degrees 3*p-1 and 5*p-1 for the flux moments."""
    weights = np.asarray(patch.weights)
    if not np.all(weights == weights.flat[0]):
        return -1
    return (3 if moment_set == "volume" else 5) * patch.u_degree - 1


def _nurbs_boundary_moments(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    center: np.ndarray,
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...] | None:
    patch = model.patches[face]
    if not isinstance(patch, BSplineSurfacePatch):
        return None
    coedges = tuple(coedge for loop in geometry.face_loops[face] for coedge in loop)
    if any(not _has_taylor_trim_source(geometry.pcurves[coedge]) for coedge in coedges):
        return None
    budget.reserve(64 * patch.control_points.size, 1024 * patch.control_points.size)
    pieces = patch.bezier_pieces()
    degree = _polynomial_moment_degree(patch, moment_set)
    totals = tuple(
        (np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64))
        for shape in _moment_shapes(moment_set)
    )
    rules: dict[tuple[int, int, bool, float, float], _GaussGreenRule] = {}
    for coedge in coedges:
        pcurve = geometry.pcurves[coedge]
        if not isinstance(pcurve, AbstractCurve):
            raise TypeError(
                "BRepMassProperties.pcurve must define an explicit source curve; got a coupled trim."
            )
        affine = affine_curve_coefficients(pcurve, np.zeros(2, dtype=np.float64))
        if affine is not None and affine[1][1] == 0:
            continue
        endpoints = geometry.edge_parameter_enclosure(
            geometry.coedge_edges[coedge], budget=budget
        )
        if np.any(endpoints[:, 0] != endpoints[:, 1]):
            return None
        cuts = _nurbs_boundary_cuts(
            patch,
            pcurve,
            float(endpoints[0, 0]),
            float(endpoints[1, 0]),
            subdivisions,
            budget,
        )
        for first, last in zip(cuts[:-1], cuts[1:], strict=True):
            values = _integrate_nurbs_boundary_span(
                model,
                geometry,
                face,
                coedge,
                center,
                pcurve,
                pieces,
                rules,
                first,
                last,
                subdivisions,
                degree,
                budget,
                moment_set,
            )
            totals = tuple(
                _array_interval_bounds(interval_add(old, value))
                for old, value in zip(totals, values, strict=True)
            )
    return totals


def _face_moment_enclosures(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    center: np.ndarray,
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    affine = _affine_face_moments(
        model, geometry, face, center, budget, moment_set=moment_set
    )
    if affine is not None:
        return affine
    boundary = _nurbs_boundary_moments(
        model, geometry, face, center, subdivisions, budget, moment_set=moment_set
    )
    if boundary is not None:
        return boundary
    return _face_cell_moment_enclosures(
        model, geometry, face, center, subdivisions, budget, moment_set=moment_set
    )


def _face_cell_moment_enclosures(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    center: np.ndarray,
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Interval integral of the Green map (q,t) -> (u0+q*(u(t)-u0),v(t)).

    The signed map handles inner loops without a sampled trim mask. Constant
    bounds on each mapped cell are rigorous quadrature bounds of order zero.
    """
    closed = _closed_round_face(
        model, geometry, face, center, budget, moment_set=moment_set
    )
    if closed is not None:
        return closed
    cells = subdivisions**2 * sum(len(loop) for loop in geometry.face_loops[face])
    budget.reserve(1024 * cells, 4096 * cells)
    totals = tuple(
        (np.zeros(shape), np.zeros(shape)) for shape in _moment_shapes(moment_set)
    )
    u0 = float(model.parameter_bounds[face, 0, 0])
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            endpoints = geometry.edge_parameter_enclosure(edge, budget=budget)
            pcurve = geometry.pcurves[coedge]
            span = interval_subtract(
                (endpoints[1, 0], endpoints[1, 1]),
                (endpoints[0, 0], endpoints[0, 1]),
            )
            cell_area = interval_divide(
                span, (np.float64(subdivisions**2), np.float64(subdivisions**2))
            )
            for cell in range(subdivisions):
                fraction = (
                    max(
                        np.float64(0),
                        np.nextafter(np.float64(cell / subdivisions), -np.inf),
                    ),
                    min(
                        np.float64(1),
                        np.nextafter(np.float64((cell + 1) / subdivisions), np.inf),
                    ),
                )
                parameter = interval_add(
                    (endpoints[0, 0], endpoints[0, 1]),
                    interval_multiply(span, fraction),
                )
                t0, t1 = float(parameter[0]), float(parameter[1])
                uv = pcurve.bounding_box(t0, t1, budget=budget)
                derivative = pcurve.derivative_bounds(t0, t1, budget=budget)
                du = _array_interval_bounds(
                    interval_subtract(
                        (uv[0, 0], uv[1, 0]), (np.float64(u0), np.float64(u0))
                    )
                )
                dv = (derivative[0][1], derivative[1][1])
                jacobian = _array_interval_bounds(interval_multiply(du, dv))
                jacobian = _array_interval_bounds(
                    interval_multiply(
                        jacobian,
                        (
                            np.float64(geometry.coedge_senses[coedge]),
                            np.float64(geometry.coedge_senses[coedge]),
                        ),
                    )
                )
                for q in range(subdivisions):
                    fraction = (
                        np.nextafter(np.float64(q / subdivisions), -np.inf),
                        np.nextafter(np.float64((q + 1) / subdivisions), np.inf),
                    )
                    mapped_u = _array_interval_bounds(
                        interval_add(
                            (np.float64(u0), np.float64(u0)),
                            _array_interval_bounds(interval_multiply(du, fraction)),
                        )
                    )
                    box = np.asarray(
                        ((mapped_u[0], uv[0, 1]), (mapped_u[1], uv[1, 1])),
                        dtype=np.float64,
                    )
                    values = _mass_integrands(
                        model.patches[face],
                        box,
                        center,
                        float(model.orientation[face]),
                        budget,
                        moment_set=moment_set,
                    )
                    weight = _array_interval_bounds(
                        interval_multiply(jacobian, cell_area)
                    )
                    totals = tuple(
                        _array_interval_bounds(
                            interval_add(
                                total,
                                _array_interval_bounds(interval_multiply(value, weight)),
                            )
                        )
                        for total, value in zip(totals, values, strict=True)
                    )
    return totals


def _central_inertia(
    volume: tuple[np.ndarray, np.ndarray],
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    centroid = _array_interval_bounds(interval_divide(first, volume))
    correction = _array_interval_bounds(
        interval_divide(
            _array_interval_bounds(
                interval_multiply(
                    (first[0][:, None], first[1][:, None]),
                    (first[0][None, :], first[1][None, :]),
                )
            ),
            volume,
        )
    )
    central = _array_interval_bounds(interval_subtract(second, correction))
    trace = (np.asarray(0.0), np.asarray(0.0))
    for axis in range(3):
        trace = _array_interval_bounds(
            interval_add(trace, (central[0][axis, axis], central[1][axis, axis]))
        )
    inertia = _array_interval_bounds(
        interval_subtract(
            _array_interval_bounds(interval_multiply((np.eye(3), np.eye(3)), trace)),
            central,
        )
    )
    return centroid, inertia


def _validate_density(
    model: BRepModel,
    density: float | None,
    density_unit: UnitDefinition | None,
    coordinate_unit: UnitDefinition | None,
    /,
) -> None:
    if density is None:
        if density_unit is not None or coordinate_unit is not None:
            raise ValueError(
                "BRepMassProperties.density must be explicit with units; got density=None."
            )
    else:
        if not np.isfinite(density) or density <= 0:
            raise ValueError(
                f"BRepMassProperties.density must be finite and positive; got {density}."
            )
        if density_unit is None or coordinate_unit is None:
            raise ValueError(
                "BRepMassProperties.density_units must include density and coordinate units; got a missing unit."
            )
        if (
            coordinate_unit.dimension != LENGTH
            or density_unit.dimension != MASS / LENGTH**3
        ):
            raise ValueError(
                "BRepMassProperties.density_units must have mass per length cubed and length dimensions; got incompatible dimensions."
            )
        if density_unit.reference_system_id != coordinate_unit.reference_system_id:
            raise ValueError(
                "BRepMassProperties.density_units must share a reference system; got different reference systems."
            )
        if density_unit.scale_to_reference * coordinate_unit.scale_to_reference**3 != 1:
            raise ValueError(
                "BRepMassProperties.density must be explicitly converted to the source coordinate unit; got a different unit scale."
            )
        if coordinate_unit != model.coordinate_contract.length_unit:
            raise ValueError(
                "BRepMassProperties.coordinate_unit must match the source coordinate contract; got a different unit identity."
            )


def _mass_record(
    model: BRepModel,
    moments: tuple[np.ndarray, ...],
    faces: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
    density: float | None,
    density_unit: UnitDefinition | None,
    /,
) -> BRepMassProperties:
    volumes, firsts, seconds, centroids, inertias = moments
    face_centroids = []
    for entry in faces:
        if entry[3][0] <= 0:
            face_centroids.append((np.full(3, -np.inf), np.full(3, np.inf)))
        else:
            face_centroids.append(
                _array_interval_bounds(interval_divide(entry[4], entry[3]))
            )
    volume_bounds = volumes
    centroid_bounds = centroids
    inertia_bounds = inertias
    return BRepMassProperties(
        model_id=model.model_id,
        volume=jnp.asarray(np.mean(volume_bounds, axis=1)),
        volume_enclosures=jnp.asarray(volume_bounds),
        first_moment_enclosures=jnp.asarray(np.asarray(firsts)),
        second_moment_enclosures=jnp.asarray(np.asarray(seconds)),
        centroid=jnp.asarray(np.mean(centroid_bounds, axis=1)),
        centroid_enclosures=jnp.asarray(centroid_bounds),
        inertia=jnp.asarray(np.mean(inertia_bounds, axis=1)),
        inertia_enclosures=jnp.asarray(inertia_bounds),
        mass=None
        if density is None
        else jnp.asarray(np.mean(volume_bounds, axis=1) * density),
        mass_enclosures=None
        if density is None
        else jnp.asarray(
            np.stack(
                _array_interval_bounds(
                    interval_multiply(
                        (
                            np.asarray(volume_bounds)[:, 0],
                            np.asarray(volume_bounds)[:, 1],
                        ),
                        (np.float64(density), np.float64(density)),
                    )
                ),
                axis=1,
            )
        ),
        mass_inertia=None
        if density is None
        else jnp.asarray(np.mean(inertia_bounds, axis=1) * density),
        mass_inertia_enclosures=None
        if density is None
        else jnp.asarray(
            np.stack(
                _array_interval_bounds(
                    interval_multiply(
                        (
                            np.asarray(inertia_bounds)[:, 0],
                            np.asarray(inertia_bounds)[:, 1],
                        ),
                        (np.float64(density), np.float64(density)),
                    )
                ),
                axis=1,
            )
        ),
        density_unit=density_unit,
        coordinate_unit=model.coordinate_contract.length_unit,
        face_area_enclosures=jnp.asarray(np.asarray([entry[3] for entry in faces])),
        face_centroid_enclosures=jnp.asarray(np.asarray(face_centroids)),
    )


@checked
def brep_mass_properties(
    model: BRepModel,
    geometry: BRepGeometry,
    /,
    *,
    subdivisions: int,
    coordinate_budget: CoordinateEnclosureBudget,
    density: float | None = None,
    density_unit: UnitDefinition | None = None,
    coordinate_unit: UnitDefinition | None = None,
) -> BRepMassProperties:
    """Bound volume, first/second moments and centroidal geometric inertia.

    Explicit cell subdivisions are a resource budget, never an accuracy promise.
    Refuses a solid whenever its volume enclosure includes zero. Native analytic
    and NURBS derivative enclosures share the same signed trim integration.
    """
    if not x64_enabled():
        raise ValueError(
            "BRepMassProperties.precision must retain binary64 enclosures; got JAX x64 disabled."
        )
    subdivisions = positive_integer(subdivisions, "BRepMassProperties.subdivisions")
    if not model.topology.solid_faces:
        raise ValueError(
            "BRepMassProperties.solids must include a closed solid; got no solids."
        )
    if model.geometry is not geometry:
        raise ValueError(
            "BRepMassProperties.geometry must belong to the model; got a different geometry."
        )
    _validate_density(model, density, density_unit, coordinate_unit)
    center, faces = _prepare_signed_face_moments(
        model, geometry, subdivisions, coordinate_budget, moment_set="mass_properties"
    )
    volumes, firsts, seconds, centroids, inertias = [], [], [], [], []
    for indices, signs in zip(
        model.topology.solid_faces, model.topology.solid_face_orientations, strict=True
    ):
        volume, first, second = _sum_signed_face_moments(
            faces, indices, signs, moment_set="mass_properties"
        )
        _check_axis_closure(faces, indices, signs)
        if volume[0] <= 0 <= volume[1]:
            raise ValueError(
                "BRepMassProperties.volume must exclude zero; got volume enclosure containing zero."
            )
        if volume[1] < 0:
            raise ValueError(
                "BRepMassProperties.volume must be positive; got a negatively oriented solid."
            )
        centroid, inertia = _central_inertia(volume, first, second)
        centroid = _array_interval_bounds(interval_add(centroid, (center, center)))
        shifted_first = _array_interval_bounds(
            interval_add(
                first, _array_interval_bounds(interval_multiply(volume, (center, center)))
            )
        )
        cross = _array_interval_bounds(
            interval_multiply(
                (center[:, None], center[:, None]), (first[0][None, :], first[1][None, :])
            )
        )
        shifted_second = _array_interval_bounds(
            interval_add(second, interval_add(cross, (cross[0].T, cross[1].T)))
        )
        center_outer = interval_multiply(
            (center[:, None], center[:, None]), (center[None, :], center[None, :])
        )
        shifted_second = _array_interval_bounds(
            interval_add(
                shifted_second,
                interval_multiply(volume, center_outer),
            )
        )
        volumes.append(volume)
        firsts.append(shifted_first)
        seconds.append(shifted_second)
        centroids.append(centroid)
        inertias.append(inertia)
    return _mass_record(
        model,
        tuple(
            np.asarray(entry) for entry in (volumes, firsts, seconds, centroids, inertias)
        ),
        faces,
        density,
        density_unit,
    )


def _prepare_signed_face_moments(
    model: BRepModel,
    geometry: BRepGeometry,
    subdivisions: int,
    coordinate_budget: CoordinateEnclosureBudget,
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[np.ndarray, tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]]:
    moment_set = parse(moment_set, _MomentSet, "BRepMeasure.moment_set")
    coordinate_budget.reserve(len(model.patches) * 16, len(model.patches) * 1024)
    boxes = np.asarray(
        [
            patch.bounding_box(
                np.asarray(model.parameter_bounds)[face], budget=coordinate_budget
            )
            for face, patch in enumerate(model.patches)
        ]
    )
    center = np.min(boxes[:, 0], axis=0) / 2 + np.max(boxes[:, 1], axis=0) / 2
    faces = tuple(
        _face_moment_enclosures(
            model,
            geometry,
            face,
            center,
            subdivisions,
            coordinate_budget,
            moment_set=moment_set,
        )
        for face in range(len(model.patches))
    )
    return center, faces


def _check_axis_closure(
    faces: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
    indices: tuple[int, ...],
    signs: tuple[int, ...],
    /,
) -> None:
    """Disjoint axis-flux volumes certify that a purported solid is open."""
    total = (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    for face, sign in zip(indices, signs, strict=True):
        total = _array_interval_bounds(
            interval_add(
                total, interval_multiply(faces[face][-1], _scalar_interval(sign))
            )
        )
    if np.max(total[0]) > np.min(total[1]):
        evidence = np.stack(total, axis=1).tolist()
        raise ValueError(
            "BRepMassProperties.closure must have overlapping three-axis volume bounds; "
            f"got disjoint axis enclosures={evidence!r}"
        )


def _sum_signed_face_moments(
    faces: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
    indices: tuple[int, ...],
    signs: tuple[int, ...],
    /,
    *,
    moment_set: _MomentSet,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    shapes = _moment_shapes(moment_set)[: 1 if moment_set == "volume" else 3]
    total = tuple((np.zeros(shape), np.zeros(shape)) for shape in shapes)
    for face, sign in zip(indices, signs, strict=True):
        total = tuple(
            _array_interval_bounds(
                interval_add(
                    old,
                    _array_interval_bounds(
                        interval_multiply(value, (np.float64(sign), np.float64(sign)))
                    ),
                )
            )
            for old, value in zip(total, faces[face][: len(total)], strict=True)
        )
    return total


@final
class BRepSignedShellVolumeEnclosures(StrictModule):
    """Signed boundary-integral volume intervals in source shell order.

    Shell-side incidence signs are retained: a correctly oriented cavity has
    negative volume. Negative and zero-containing intervals are both published.
    """

    model_id: str = eqx.field(static=True)
    shell_ids: tuple[BRepEntityId, ...] = eqx.field(static=True)
    volume_enclosures: Array


@checked
def brep_signed_shell_volume_enclosures(
    model: BRepModel,
    geometry: BRepGeometry,
    /,
    *,
    subdivisions: int,
    coordinate_budget: CoordinateEnclosureBudget,
) -> BRepSignedShellVolumeEnclosures:
    """Publish signed shell bounds without the solid mass-property positivity gate."""
    if not x64_enabled():
        raise ValueError(
            "BRepSignedShellVolumeEnclosures.precision must retain binary64 enclosures; got JAX x64 disabled."
        )
    subdivisions = positive_integer(
        subdivisions, "BRepSignedShellVolumeEnclosures.subdivisions"
    )
    if model.geometry is not geometry:
        raise ValueError(
            "BRepSignedShellVolumeEnclosures.geometry must belong to the model; got a different geometry."
        )
    coordinate_budget.reserve(
        8 * max(1, len(geometry.shell_faces)), 128 * max(1, len(geometry.shell_faces))
    )
    if geometry.shell_faces:
        _, faces = _prepare_signed_face_moments(
            model, geometry, subdivisions, coordinate_budget, moment_set="volume"
        )
        volumes = np.asarray(
            [
                _sum_signed_face_moments(faces, indices, signs, moment_set="volume")[0]
                for indices, signs in zip(
                    geometry.shell_faces, geometry.shell_orientations, strict=True
                )
            ],
            dtype=np.float64,
        )
    else:
        volumes = np.empty((0, 2), dtype=np.float64)
    return BRepSignedShellVolumeEnclosures(
        model_id=model.model_id,
        shell_ids=tuple(
            BRepEntityId(model.source_revision, "shell", index)
            for index in range(len(geometry.shell_faces))
        ),
        volume_enclosures=jnp.asarray(volumes),
    )


class QuadraturePolicy(Protocol):
    """Resolution and allowance metadata consumed by trim quadrature."""

    @property
    def quadrature_order(self) -> int: ...
    @property
    def quadrature_subdivisions(self) -> int: ...
    @property
    def maximum_operations(self) -> int: ...
    @property
    def maximum_points(self) -> int: ...
    @property
    def maximum_scratch_bytes(self) -> int: ...


@final
class BRepMeasureResult(StrictModule):
    """Face areas and solid volumes with fine-minus-coarse error estimates.

    The differences estimate quadrature error; they are not certified bounds.
    """

    face_areas: Array
    face_area_error_estimates: Array
    solid_volumes: Array
    solid_volume_error_estimates: Array


@final
class BRepQuadrature(StrictModule):
    """Trimmed-face nodes, oriented vector areas, and signed quadrature weights."""

    points: Array
    vector_areas: Array
    areas: Array
    parameters: Array
    weights: Array
    faces: Array


@dataclass(frozen=True, slots=True)
class _FaceQuadrature:
    parameters: np.ndarray
    weights: np.ndarray
    points: np.ndarray
    vector_areas: np.ndarray
    areas: np.ndarray


def _gauss(order: int, /) -> tuple[np.ndarray, np.ndarray]:
    rule = gauss_legendre_data(order)
    return np.asarray(rule.nodes, dtype=np.float64), np.asarray(
        rule.weights, dtype=np.float64
    )


def _coedge_quadrature_breaks(
    pcurve: BRepPCurve,
    first: float,
    last: float,
    subdivisions: int,
    budget: BRepQueryBudget,
    /,
) -> np.ndarray:
    """Every source chart/knot interval is integrated separately before rule refinement."""
    source = pcurve.source_curve if isinstance(pcurve, AffinePCurve) else pcurve
    visits = (
        source.curve.num_charts + 1
        if isinstance(source, IntersectionPCurve)
        else source.knots.size
        if isinstance(source, BSplineCurve)
        else 0
    )
    scratch = 160 * (visits + 2) + 96 * budget.points
    budget.admit(visits, 0, scratch)
    budget.consume(visits)
    boundaries = [first, last]
    if isinstance(source, IntersectionPCurve):
        for chart in range(source.curve.num_charts + 1):
            parameter = (
                source.first + source.last - chart if source.reversed else float(chart)
            )
            if first < parameter < last:
                boundaries.append(parameter)
    elif isinstance(source, BSplineCurve):
        boundaries.extend(
            float(value) for value in np.asarray(source.knots) if first < value < last
        )
    ordered = sorted(set(boundaries))
    count = (len(ordered) - 1) * subdivisions + 1
    budget.admit(count, 0, scratch + 24 * count)
    budget.consume(count)
    return np.concatenate(
        [
            np.linspace(start, end, subdivisions + 1, dtype=np.float64)[:-1]
            for start, end in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        + [np.asarray([last], dtype=np.float64)]
    )


@checked
def surface_quadrature_u_breaks(
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    budget: BRepQueryBudget,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Native source chart walls, without materializing tensor-product pieces."""
    budget.admit(0, 0, 128)
    cuts = {float(box[0, 0]), float(box[1, 0])}
    scratch = 128

    def knots(values: Array, first: float, last: float) -> None:
        nonlocal scratch
        count = values.size
        # Source visits and simultaneous host leaf/set/publication storage,
        # admitted before transferring or growing any source-derived payload.
        scratch += 160 * count
        budget.admit(count, 0, scratch)
        budget.consume(count)
        for value in np.asarray(values):
            if first < value < last:
                cuts.add(float(value))

    def curve_walls(
        curve: AbstractCurve | IntersectionCurve,
        first: float,
        last: float,
        *,
        coordinate_budget: CoordinateEnclosureBudget = coordinate_budget,
    ) -> None:
        nonlocal scratch
        budget.admit(1, 0, scratch)
        budget.consume(1)
        if isinstance(curve, PlacedCurve):
            curve_walls(
                curve.definition, first, last, coordinate_budget=coordinate_budget
            )
        elif isinstance(curve, OffsetCurve):
            curve_walls(curve.base, first, last, coordinate_budget=coordinate_budget)
        elif isinstance(curve, SurfaceIsoparametricCurve):
            surface_walls(
                curve.surface,
                curve.parameter_box(first, last, budget=coordinate_budget),
                1 - curve.fixed_axis,
                coordinate_budget=coordinate_budget,
            )
        elif isinstance(curve, BSplineCurve):
            knots(curve.knots, first, last)
        elif isinstance(curve, IntersectionCurve):
            count = curve.num_charts + 1
            scratch += 160 * count
            budget.admit(count, 0, scratch)
            budget.consume(count)
            for chart in range(count):
                if first < chart < last:
                    cuts.add(float(chart))

    def surface_walls(
        source: AbstractSurfacePatch,
        bounds: np.ndarray,
        axis: int,
        *,
        coordinate_budget: CoordinateEnclosureBudget = coordinate_budget,
    ) -> None:
        budget.admit(1, 0, scratch)
        budget.consume(1)
        if isinstance(source, PlacedSurface):
            surface_walls(
                source.definition, bounds, axis, coordinate_budget=coordinate_budget
            )
        elif isinstance(source, OffsetSurface):
            surface_walls(source.base, bounds, axis, coordinate_budget=coordinate_budget)
        elif isinstance(source, ExtrusionSurface) and axis == 0:
            curve_walls(
                source.curve,
                float(bounds[0, 0]),
                float(bounds[1, 0]),
                coordinate_budget=coordinate_budget,
            )
        elif isinstance(source, RevolutionSurface) and axis == 1:
            curve_walls(
                source.curve,
                float(bounds[0, 1]),
                float(bounds[1, 1]),
                coordinate_budget=coordinate_budget,
            )
        elif isinstance(source, RuledSurface) and axis == 0:
            first, last = float(bounds[0, 0]), float(bounds[1, 0])
            curve_walls(source.first, first, last, coordinate_budget=coordinate_budget)
            curve_walls(source.second, first, last, coordinate_budget=coordinate_budget)
        elif isinstance(source, BSplineSurfacePatch):
            knots(
                source.u_knots if axis == 0 else source.v_knots,
                float(bounds[0, axis]),
                float(bounds[1, axis]),
            )

    surface_walls(patch, box, 0, coordinate_budget=coordinate_budget)
    return np.asarray(sorted(cuts), dtype=np.float64)


@checked
def green_nodes(
    geometry: BRepGeometry,
    face: int,
    box: np.ndarray,
    order: int,
    subdivisions: int,
    u_breaks: np.ndarray,
    budget: BRepQueryBudget,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameter nodes and weights with ``sum w g(u, v) = int_D g du dv``."""
    nodes, weights = _gauss(order)
    ranges = np.asarray(geometry.edge_ranges)
    lower_u = float(box[0, 0])
    parameters, node_weights = [], []
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            first, last = ranges[edge]
            pcurve = geometry.pcurves[coedge]
            breaks = _coedge_quadrature_breaks(
                pcurve, float(first), float(last), subdivisions, budget
            )
            outer_count = (breaks.size - 1) * nodes.size
            budget.admit(
                outer_count,
                outer_count,
                96 * budget.points + 128 * outer_count,
            )
            budget.consume(outer_count, points=outer_count)
            half = 0.5 * (breaks[1:] - breaks[:-1])
            t = (0.5 * (breaks[1:] + breaks[:-1]))[:, None] + half[:, None] * nodes
            outer_weights = (half[:, None] * weights).reshape(-1)
            uv, velocity = curve_value_and_derivative_host(pcurve, t.reshape(-1))
            outer = geometry.coedge_senses[coedge] * outer_weights * velocity[:, 1]
            keep = outer != 0.0
            boundary_u, boundary_v, outer = uv[keep, 0], uv[keep, 1], outer[keep]
            for first_u, last_u in zip(u_breaks[:-1], u_breaks[1:], strict=True):
                lower = np.maximum(lower_u, first_u)
                upper = np.minimum(boundary_u, last_u)
                selected = upper > lower
                if not np.any(selected):
                    continue
                count = int(np.count_nonzero(selected)) * nodes.size
                scratch = 96 * (budget.points + count) + 24 * boundary_u.size
                budget.admit(count, count, scratch)
                budget.consume(count, points=count)
                span = 0.5 * (upper[selected] - lower)
                inner_u = (lower + span)[:, None] + span[:, None] * nodes
                parameters.append(
                    np.stack(
                        (
                            inner_u,
                            np.broadcast_to(boundary_v[selected, None], inner_u.shape),
                        ),
                        axis=-1,
                    ).reshape(-1, 2)
                )
                node_weights.append(
                    (outer[selected, None] * span[:, None] * weights).reshape(-1)
                )
    return np.concatenate(parameters), np.concatenate(node_weights)


@checked
def face_quadrature(
    patch: AbstractSurfacePatch,
    orientation: float,
    parameters: np.ndarray,
    weights: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, differential = surface_value_and_differential_host(patch, parameters)
    normal = np.cross(differential[:, :, 0], differential[:, :, 1])
    return (
        points,
        orientation * weights[:, None] * normal,
        weights * np.linalg.norm(normal, axis=1),
    )


@checked
def trimmed_face_rules(
    model: BRepModel,
    policy: QuadraturePolicy,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Array, Array], ...]:
    """Return per-face parameter nodes and weights integrating over source trim loops."""
    geometry = model.require_geometry()
    bounds = np.asarray(model.parameter_bounds)
    rules = []
    budget = BRepQueryBudget(
        policy.maximum_operations, policy.maximum_points, policy.maximum_scratch_bytes
    )
    for face in range(len(model.patches)):
        parameters, weights = green_nodes(
            geometry,
            face,
            bounds[face],
            policy.quadrature_order,
            policy.quadrature_subdivisions,
            surface_quadrature_u_breaks(
                model.patches[face],
                bounds[face],
                budget,
                coordinate_budget=coordinate_budget,
            ),
            budget,
        )
        rules.append((jnp.asarray(parameters), jnp.asarray(weights)))
    return tuple(rules)


@checked
def prepare_quadrature(
    model: BRepModel,
    geometry: BRepGeometry,
    policy: QuadraturePolicy,
    /,
    *,
    coordinate_budget: CoordinateEnclosureBudget,
) -> tuple[BRepQuadrature, BRepMeasureResult]:
    orientation = np.asarray(model.orientation)
    bounds = np.asarray(model.parameter_bounds)
    fine: list[_FaceQuadrature] = []
    coarse: list[_FaceQuadrature] = []
    budget = BRepQueryBudget(
        policy.maximum_operations, policy.maximum_points, policy.maximum_scratch_bytes
    )
    u_breaks = tuple(
        surface_quadrature_u_breaks(
            patch, bounds[face], budget, coordinate_budget=coordinate_budget
        )
        for face, patch in enumerate(model.patches)
    )
    for subdivisions, store in (
        (policy.quadrature_subdivisions, fine),
        (policy.quadrature_subdivisions // 2, coarse),
    ):
        for face, patch in enumerate(model.patches):
            parameters, weights = green_nodes(
                geometry,
                face,
                bounds[face],
                policy.quadrature_order,
                subdivisions,
                u_breaks[face],
                budget,
            )
            points, vector_areas, areas = face_quadrature(
                patch, float(orientation[face]), parameters, weights
            )
            store.append(
                _FaceQuadrature(
                    parameters=parameters,
                    weights=weights,
                    points=points,
                    vector_areas=vector_areas,
                    areas=areas,
                )
            )

    def measure(store: list[_FaceQuadrature], /) -> tuple[np.ndarray, np.ndarray]:
        face_areas = np.asarray([np.sum(entry.areas) for entry in store])
        face_volumes = np.asarray(
            [np.sum(entry.points * entry.vector_areas) / 3.0 for entry in store]
        )
        solid_volumes = np.asarray(
            [
                sum(
                    sign * face_volumes[face]
                    for face, sign in zip(faces, signs, strict=True)
                )
                for faces, signs in zip(
                    model.topology.solid_faces,
                    model.topology.solid_face_orientations,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        return face_areas, solid_volumes

    fine_areas, fine_volumes = measure(fine)
    coarse_areas, coarse_volumes = measure(coarse)
    quadrature = BRepQuadrature(
        points=jnp.asarray(np.concatenate([entry.points for entry in fine])),
        vector_areas=jnp.asarray(np.concatenate([entry.vector_areas for entry in fine])),
        areas=jnp.asarray(np.concatenate([entry.areas for entry in fine])),
        parameters=jnp.asarray(np.concatenate([entry.parameters for entry in fine])),
        weights=jnp.asarray(np.concatenate([entry.weights for entry in fine])),
        faces=jnp.asarray(
            np.concatenate(
                [
                    np.full(entry.parameters.shape[0], face)
                    for face, entry in enumerate(fine)
                ]
            ).astype(np.int32)
        ),
    )
    return quadrature, BRepMeasureResult(
        face_areas=jnp.asarray(fine_areas),
        face_area_error_estimates=jnp.asarray(np.abs(fine_areas - coarse_areas)),
        solid_volumes=jnp.asarray(fine_volumes),
        solid_volume_error_estimates=jnp.asarray(np.abs(fine_volumes - coarse_volumes)),
    )
