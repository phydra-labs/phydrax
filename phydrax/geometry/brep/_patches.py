#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from math import atan2, gcd, lcm, tau, ulp
from typing import cast, final, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from numpy.typing import NDArray

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import (
    bspline_jet_stencil,
    PreparedSourceSpans,
    rational_bspline_curve_spans,
    rational_bspline_surface_spans,
    RationalBSplineCurveSpan,
    RationalBSplineSurfaceSpan,
    RationalSplineJet,
    TensorBSplineJetPlan,
)
from ..._strict import StrictModule
from ..._validation import finite_real_scalar
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    coefficient_payload,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ...typing import ConvertibleToArray, Dim, Float64, Scalar
from .._interval_enclosure import Interval


type _ExactSphereRevolutionWitness = tuple[
    tuple[ExactScalar, ...],
    ExactScalar,
    tuple[Fraction, Fraction, Fraction],
    int,
]


class AbstractSurfacePatch(StrictModule):
    """Pure-JAX parametric surface patch in native CAD coordinates."""

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> Array:
        raise NotImplementedError

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        """Enclose the authoritative source map on a validated UV rectangle."""
        from ._root_bindings import _patch_box

        box = np.asarray(self.validate_parameter_box(parameter_box), dtype=np.float64)
        return _patch_box(self, box[0], box[1])

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enclose physical-coordinate Jacobian columns on a UV box.

        Exact-capable source carriers override this method. Unsupported families
        refuse rather than substituting samples or a tessellation fit.
        """
        self.validate_parameter_box(parameter_box)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Surface derivative bounds currently support order=1 only.")
        raise NotImplementedError(
            f"A continuous Jacobian enclosure is unavailable for {type(self).__name__}."
        )

    @property
    def parameter_domain(
        self, /
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return None

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return None, None

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        """Return source-declared period symbols, not numerical period estimates."""
        return None, None

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        """Report established C1 continuity, not an assumption about custom patches."""
        self.validate_parameter_box(parameter_box)
        return type(self) in (
            PlanePatch,
            CylinderPatch,
            ConePatch,
            SpherePatch,
            TorusPatch,
        )

    def is_c1_along(
        self,
        fixed_axis: int,
        fixed_value: float,
        first: float,
        last: float,
        /,
    ) -> bool:
        if isinstance(fixed_axis, bool) or not isinstance(fixed_axis, int):
            raise TypeError("fixed_axis must be an integer.")
        if fixed_axis not in (0, 1):
            raise ValueError("fixed_axis must be 0 or 1.")
        if not all(np.isfinite(value) for value in (fixed_value, first, last)):
            raise ValueError("Surface isoline parameters must be finite.")
        if first > last:
            raise ValueError("Surface isoline ranges must be nondecreasing.")
        free_axis = 1 - fixed_axis
        domain = self.parameter_domain
        if domain is not None and (
            fixed_value < domain[0][fixed_axis]
            or fixed_value > domain[1][fixed_axis]
            or first < domain[0][free_axis]
            or last > domain[1][free_axis]
        ):
            raise ValueError("Surface isoline range must lie in its source domain.")
        period = self.periods[free_axis]
        if period is not None and last - first > period * (
            1.0 + 128.0 * np.finfo(np.float64).eps
        ):
            raise ValueError("A periodic isoline interval may cover at most one period.")
        return type(self) in (
            PlanePatch,
            CylinderPatch,
            ConePatch,
            SpherePatch,
            TorusPatch,
        )

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> Array:
        box = np.asarray(parameter_box, dtype=np.float64)
        if (
            box.shape != (2, 2)
            or not np.all(np.isfinite(box))
            or np.any(box[1] <= box[0])
        ):
            raise ValueError(
                "A surface parameter box must be finite, shaped (2, 2), and increasing."
            )
        domain = self.parameter_domain
        if domain is not None:
            domain_ = np.asarray(domain, dtype=np.float64)
            if np.any(box[0] < domain_[0]) or np.any(box[1] > domain_[1]):
                raise ValueError("The parameter box must lie within the patch domain.")
        for axis, period in enumerate(self.periods):
            if period is not None and box[1, axis] - box[0, axis] > period * (
                1.0 + 128.0 * np.finfo(np.float64).eps
            ):
                raise ValueError(
                    "A periodic parameter interval may cover at most one period."
                )
        return jnp.asarray(box, dtype=jnp.float64)


class OffsetSurface(AbstractSurfacePatch):
    """Exact normal-offset operation tree over an unchanged smooth source patch."""

    base: AbstractSurfacePatch
    distance: Array
    distance_exact: Fraction = eqx.field(static=True)
    frame_sign: int = eqx.field(static=True)
    _sphere_revolution_witness: _ExactSphereRevolutionWitness | None = eqx.field(
        static=True
    )

    def __init__(
        self,
        base: AbstractSurfacePatch,
        distance: float | Fraction,
        /,
        *,
        frame_sign: int | None = None,
    ) -> None:
        if not isinstance(base, AbstractSurfacePatch):
            raise TypeError("OffsetSurface.base must be an AbstractSurfacePatch.")
        if isinstance(distance, bool):
            raise TypeError("OffsetSurface.distance must be a finite real scalar.")
        exact = (
            distance
            if isinstance(distance, Fraction)
            else Fraction.from_float(float(distance))
        )
        numeric = float(exact)
        if not np.isfinite(numeric):
            raise ValueError("OffsetSurface.distance must be finite.")
        sphere_witness = None
        computed_frame_sign = 0
        if isinstance(base, TorusPatch):
            columns = (
                base.first_axis_exact,
                base.second_axis_exact,
                base.axis_exact,
            )
            gram = tuple(
                tuple(
                    sum(
                        (
                            a * b
                            for a, b in zip(columns[first], columns[second], strict=True)
                        ),
                        Fraction(0),
                    )
                    for second in range(3)
                )
                for first in range(3)
            )
            identity = tuple(
                tuple(Fraction(int(row == column)) for column in range(3))
                for row in range(3)
            )
            if gram == identity:
                rows = tuple(
                    tuple(columns[column][row] for column in range(3)) for row in range(3)
                )
                determinant = (
                    rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
                    - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
                    + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
                )
                computed_frame_sign = (
                    1 if determinant > 0 else -1 if determinant < 0 else 0
                )
        elif isinstance(base, (CylinderPatch, ConePatch, SpherePatch)):
            from .._rational import rational_vector
            from ._exact import rational_determinant

            columns = tuple(
                rational_vector(np.asarray(axis))
                for axis in (base.first_axis, base.second_axis, base.axis)
            )
            gram = tuple(
                tuple(
                    sum(
                        (
                            a * b
                            for a, b in zip(columns[first], columns[second], strict=True)
                        ),
                        Fraction(0),
                    )
                    for second in range(3)
                )
                for first in range(3)
            )
            identity = tuple(
                tuple(Fraction(int(row == column)) for column in range(3))
                for row in range(3)
            )
            if gram == identity:
                rows = tuple(
                    tuple(columns[column][row] for column in range(3)) for row in range(3)
                )
                determinant = rational_determinant(*rows)
                computed_frame_sign = (
                    1 if determinant > 0 else -1 if determinant < 0 else 0
                )
        if isinstance(base, RevolutionSurface):
            sphere_witness = _exact_spherical_revolution_witness(base)
            if sphere_witness is not None:
                computed_frame_sign = sphere_witness[3]
        if frame_sign is not None and (
            type(frame_sign) is not int or frame_sign not in (-1, 1)
        ):
            raise ValueError("OffsetSurface.frame_sign must be -1 or 1 when declared.")
        self.base = base
        self.distance = jnp.asarray(numeric, dtype=jnp.float64)
        self.distance_exact = exact
        self.frame_sign = computed_frame_sign if frame_sign is None else frame_sign
        self._sphere_revolution_witness = (
            (
                sphere_witness[0],
                sphere_witness[1],
                sphere_witness[2],
                self.frame_sign,
            )
            if sphere_witness is not None
            else None
        )

    def _evaluate_one(self, parameters: Array, /) -> Array:
        uv = jnp.asarray(parameters, dtype=jnp.float64)
        layers: list[OffsetSurface] = []
        base: AbstractSurfacePatch = self
        while type(base) is OffsetSurface:
            layers.append(base)
            base = base.base
        if type(base) is PlanePatch:
            point = base.evaluate(uv)
            normal = jnp.cross(base.first_axis, base.second_axis)
            normal = normal / jnp.linalg.norm(normal)
            distances = jnp.stack(
                tuple(
                    layer.distance * (layer.frame_sign if layer.frame_sign else 1)
                    for layer in reversed(layers)
                )
            )

            def apply_offset(value: Array, distance: Array) -> tuple[Array, None]:
                return value + distance * normal, None

            return jax.lax.scan(apply_offset, point, distances)[0]
        if isinstance(self.base, SpherePatch) and self.frame_sign != 0:
            longitude, latitude = uv[0], uv[1]
            equatorial = (
                jnp.cos(longitude) * self.base.first_axis
                + jnp.sin(longitude) * self.base.second_axis
            )
            direction = (
                jnp.cos(latitude) * equatorial + jnp.sin(latitude) * self.base.axis
            )
            return (
                self.base.center
                + (self.base.radius + self.frame_sign * self.distance) * direction
            )
        if self._sphere_revolution_witness is not None:
            base = self.base
            if not isinstance(base, RevolutionSurface) or not isinstance(
                base.curve, CircleCurve
            ):
                raise RuntimeError(
                    "OffsetSurface lost its exact spherical revolution source."
                )
            point = base.evaluate(uv)
            radial = (point - base.curve.center) / base.curve.radius
            return (
                base.curve.center
                + (base.curve.radius + self.frame_sign * self.distance) * radial
            )
        if isinstance(self.base, CylinderPatch) and self.frame_sign != 0:
            angle, height = uv[0], uv[1]
            radial = (
                jnp.cos(angle) * self.base.first_axis
                + jnp.sin(angle) * self.base.second_axis
            )
            return (
                self.base.origin
                + (self.base.radius + self.frame_sign * self.distance) * radial
                + height * self.base.axis
            )
        if isinstance(self.base, TorusPatch) and self.frame_sign != 0:
            longitude, tube_angle = uv[0], uv[1]
            radial = (
                jnp.cos(longitude) * self.base.first_axis
                + jnp.sin(longitude) * self.base.second_axis
            )
            normal = (
                jnp.cos(tube_angle) * radial + jnp.sin(tube_angle) * self.base.axis
            ) * self.frame_sign
            return self.base.evaluate(uv) + self.distance * normal
        if isinstance(self.base, ConePatch) and self.frame_sign != 0:
            angle = uv[0]
            radial = (
                jnp.cos(angle) * self.base.first_axis
                + jnp.sin(angle) * self.base.second_axis
            )
            tangent = jnp.tan(self.base.semi_angle)
            normal = (radial - tangent * self.base.axis) / jnp.sqrt(
                1.0 + tangent * tangent
            )
            return self.base.evaluate(uv) + self.frame_sign * self.distance * normal
        point = self.base.evaluate(uv)
        differential = jax.jacfwd(self.base.evaluate)(uv)
        normal = jnp.cross(differential[:, 0], differential[:, 1])
        orientation = self.frame_sign if self.frame_sign != 0 else 1
        return point + self.distance * orientation * normal / jnp.linalg.norm(normal)

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim == 1:
            return self._evaluate_one(values)
        leading = values.shape[:-1]
        return jax.vmap(self._evaluate_one)(values.reshape((-1, 2))).reshape(
            (*leading, 3)
        )

    @property
    def parameter_domain(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return self.base.parameter_domain

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.base.periods

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.base.native_period_symbols()

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> Array:
        return self.base.validate_parameter_box(parameter_box)

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        return self.base.is_c1_on(parameter_box)

    def exact_sphere_revolution_witness(self, /) -> _ExactSphereRevolutionWitness | None:
        """Return exact source sphere data retained by a revolution offset."""
        return self._sphere_revolution_witness

    def analytic_equivalent(self, /) -> AbstractSurfacePatch | None:
        """Lower only when an existing analytic carrier has exactly the same map."""
        if self.frame_sign == 0:
            return None
        distance = self.distance_exact
        if isinstance(self.base, SpherePatch):
            radius = Fraction(float(self.base.radius)) + self.frame_sign * distance
            if radius > 0 and Fraction.from_float(float(radius)) == radius:
                return SpherePatch(
                    self.base.center,
                    self.base.first_axis,
                    self.base.second_axis,
                    self.base.axis,
                    float(radius),
                )
        if isinstance(self.base, CylinderPatch):
            radius = Fraction(float(self.base.radius)) + self.frame_sign * distance
            if radius > 0 and Fraction.from_float(float(radius)) == radius:
                return CylinderPatch(
                    self.base.origin,
                    self.base.first_axis,
                    self.base.second_axis,
                    self.base.axis,
                    float(radius),
                )
        if isinstance(self.base, TorusPatch):
            major = self.base.major_radius_exact
            minor = self.base.minor_radius_exact + self.frame_sign * distance
            if not 0 < minor < major:
                return None
            try:
                major_numeric, minor_numeric = float(major), float(minor)
                base_major_numeric = float(np.asarray(self.base.major_radius))
                base_minor_numeric = float(np.asarray(self.base.minor_radius))
                distance_numeric = float(np.asarray(self.distance))
                exact_shadows_match = (
                    major_numeric == base_major_numeric
                    and float(self.base.major_radius_exact) == base_major_numeric
                    and float(self.base.minor_radius_exact) == base_minor_numeric
                    and float(distance) == distance_numeric
                    and np.array_equal(
                        np.asarray(self.base.center, dtype=np.float64),
                        np.asarray(
                            tuple(float(value) for value in self.base.center_exact),
                            dtype=np.float64,
                        ),
                    )
                    and all(
                        np.array_equal(
                            np.asarray(numeric, dtype=np.float64),
                            np.asarray(tuple(float(value) for value in exact)),
                        )
                        for numeric, exact in (
                            (self.base.first_axis, self.base.first_axis_exact),
                            (self.base.second_axis, self.base.second_axis_exact),
                            (self.base.axis, self.base.axis_exact),
                        )
                    )
                )
            except (OverflowError, TypeError, ValueError):
                return None
            if (
                not np.isfinite(major_numeric)
                or not np.isfinite(minor_numeric)
                or not exact_shadows_match
            ):
                return None
            return TorusPatch(
                self.base.center,
                self.base.first_axis,
                self.base.second_axis,
                self.base.axis,
                major_numeric,
                minor_numeric,
                center_exact=self.base.center_exact,
                first_axis_exact=self.base.first_axis_exact,
                second_axis_exact=self.base.second_axis_exact,
                axis_exact=self.base.axis_exact,
                major_radius_exact=major,
                minor_radius_exact=minor,
            )
        return None

    def identity_payload(self) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "normal-offset-surface",
            "base": carrier_payload(self.base),
            "distance": (self.distance_exact.numerator, self.distance_exact.denominator),
            "frame_sign": self.frame_sign,
        }


def _curve_vector(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] not in (2, 3):
        raise ValueError(f"{name} must be a two- or three-dimensional vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite.")
    return vector


def _orthonormal_pair(
    first_axis: np.ndarray, second_axis: np.ndarray, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    if first_axis.shape != second_axis.shape:
        raise ValueError(f"{name} curve frame axes must have matching dimensions.")
    tolerance = 128.0 * np.finfo(np.float64).eps
    if (
        abs(float(np.linalg.norm(first_axis)) - 1.0) > tolerance
        or abs(float(np.linalg.norm(second_axis)) - 1.0) > tolerance
        or abs(float(first_axis @ second_axis)) > tolerance
    ):
        raise ValueError(f"{name} curve frame axes must be orthonormal.")
    return first_axis, second_axis


class AbstractCurve(StrictModule):
    """Native parametric curve used by B-Rep edges and p-curves."""

    @property
    @abstractmethod
    def ambient_dimension(self) -> int:
        raise NotImplementedError

    @property
    def period(self) -> float | None:
        return None

    def native_period_symbol(self, /) -> str | None:
        """Return a mathematical source period symbol, if one is declared."""
        return None

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return None

    def is_c1_on(self, first: float, last: float, /) -> bool:
        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("Curve ranges must be finite and nondecreasing.")
        domain = self.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            raise ValueError("Curve range must lie in its source parameter domain.")
        return type(self) in (
            LineCurve,
            ExactLineCurve,
            CircleCurve,
            EllipseCurve,
            ParabolaCurve,
            HyperbolaCurve,
            BSplineCurve,
        )

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        self.is_c1_on(first, last)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Curve derivative bounds support order=1 only.")
        raise NotImplementedError(
            f"A continuous derivative enclosure is unavailable for {type(self).__name__}."
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        """Enclose the represented source map, without substituting samples."""
        from .._interval_enclosure import prepare_interval_function

        AbstractCurve.is_c1_on(self, first, last)

        def value(parameter: Array, /) -> Array:
            return self.evaluate(parameter[0])

        prepared = prepare_interval_function(value, 1)
        lower, upper = prepared.evaluate(
            np.asarray(((first,),), dtype=np.float64),
            np.asarray(((last,),), dtype=np.float64),
        )
        return np.stack((lower[0], upper[0]))

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> Array:
        raise NotImplementedError


class LineCurve(AbstractCurve):
    """Affine line ``origin + t * direction`` with exact source coefficients."""

    origin: Array
    direction: Array
    origin_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    direction_exact: tuple[ExactScalar, ...] = eqx.field(static=True)

    def __init__(
        self,
        origin: ConvertibleToArray,
        direction: ConvertibleToArray,
        *,
        origin_exact: tuple[ExactScalar, ...] | None = None,
        direction_exact: tuple[ExactScalar, ...] | None = None,
    ) -> None:
        origin_ = _curve_vector(origin, "origin")
        direction_ = _curve_vector(direction, "direction")
        if origin_.shape != direction_.shape:
            raise ValueError("LineCurve.direction must match the origin dimension.")
        if not np.any(direction_ != 0.0):
            raise ValueError("LineCurve.direction must be nonzero.")

        def exact_vector(
            values: tuple[ExactScalar, ...] | None, shadow: np.ndarray, name: str, /
        ) -> tuple[ExactScalar, ...]:
            result = (
                tuple(Fraction.from_float(float(value)) for value in shadow)
                if values is None
                else tuple(values)
            )
            if len(result) != shadow.size or any(
                not isinstance(value, (Fraction, RadicalElement)) for value in result
            ):
                raise TypeError(f"LineCurve.{name} must contain exact source scalars.")
            if not np.array_equal(
                np.asarray(tuple(float(value) for value in result), dtype=np.float64),
                shadow,
            ):
                raise ValueError(f"LineCurve.{name} shadow must round its exact source.")
            return result

        self.origin = jnp.asarray(origin_, dtype=jnp.float64)
        self.direction = jnp.asarray(direction_, dtype=jnp.float64)
        self.origin_exact = exact_vector(origin_exact, origin_, "origin_exact")
        self.direction_exact = exact_vector(
            direction_exact, direction_, "direction_exact"
        )

    @property
    def ambient_dimension(self) -> int:
        return self.origin.shape[0]

    def evaluate(self, parameters: Array, /) -> Array:
        parameter = jnp.asarray(parameters, dtype=self.origin.dtype)
        return self.origin + parameter[..., None] * self.direction

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        self.is_c1_on(first, last)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Line derivative bounds support order=1 only.")
        from .._rational import fraction_down, fraction_up

        bounds = tuple(
            coefficient_interval(value, bits=128) for value in self.direction_exact
        )
        return (
            np.asarray(
                tuple(fraction_down(lower) for lower, _ in bounds), dtype=np.float64
            ),
            np.asarray(
                tuple(fraction_up(upper) for _, upper in bounds), dtype=np.float64
            ),
        )

    def evaluate_exact(self, parameter: Fraction | float, /) -> tuple[ExactScalar, ...]:
        value = (
            parameter
            if isinstance(parameter, Fraction)
            else Fraction.from_float(float(parameter))
        )
        return tuple(
            origin + value * direction
            for origin, direction in zip(
                self.origin_exact, self.direction_exact, strict=True
            )
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._rational import fraction_down, fraction_up

        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("LineCurve bounds must be finite and ordered.")
        first_exact = Fraction.from_float(float(first))
        last_exact = Fraction.from_float(float(last))
        lower: list[float] = []
        upper: list[float] = []
        for origin, direction in zip(
            self.origin_exact, self.direction_exact, strict=True
        ):
            left = origin + first_exact * direction
            right = origin + last_exact * direction
            low = left if left <= right else right
            high = right if left <= right else left
            low_interval = coefficient_interval(low)
            high_interval = coefficient_interval(high)
            lower.append(fraction_down(low_interval[0]))
            upper.append(fraction_up(high_interval[1]))
        return np.stack((np.asarray(lower), np.asarray(upper)))

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "exact-affine-line-curve",
            "origin": tuple(coefficient_payload(value) for value in self.origin_exact),
            "direction": tuple(
                coefficient_payload(value) for value in self.direction_exact
            ),
        }


class ExactLineCurve(AbstractCurve):
    """Affine source segment with rational endpoints and float JAX realization."""

    first_exact: tuple[Fraction, ...] = eqx.field(static=True)
    last_exact: tuple[Fraction, ...] = eqx.field(static=True)
    first: Array
    last: Array

    def __init__(
        self,
        first_exact: ConvertibleToArray | Sequence[Fraction],
        last_exact: ConvertibleToArray | Sequence[Fraction],
    ) -> None:
        def exact_point(
            value: ConvertibleToArray | Sequence[Fraction], name: str, /
        ) -> tuple[Fraction, ...]:
            components = np.asarray(value, dtype=np.dtype("O"))
            if components.ndim == 0:
                raise TypeError(f"{name} must be a two- or three-dimensional point.")
            if components.ndim != 1 or components.size not in (2, 3):
                raise ValueError(f"{name} must be a two- or three-dimensional point.")
            result: list[Fraction] = []
            for index in range(components.size):
                component: object = components[index]
                if isinstance(component, bool) or isinstance(component, np.bool_):
                    raise TypeError(f"{name} coordinates must be finite real scalars.")
                if isinstance(component, Fraction):
                    result.append(component)
                elif isinstance(component, (int, np.integer)):
                    result.append(Fraction(int(component)))
                elif isinstance(component, (float, np.floating)) and np.isfinite(
                    component
                ):
                    result.append(Fraction.from_float(float(component)))
                else:
                    raise TypeError(f"{name} coordinates must be finite real scalars.")
            return tuple(result)

        first_value = exact_point(first_exact, "first_exact")
        last_value = exact_point(last_exact, "last_exact")
        if len(first_value) != len(last_value):
            raise ValueError("ExactLineCurve endpoints must have matching dimensions.")
        if first_value == last_value:
            raise ValueError("ExactLineCurve endpoints must be distinct.")
        self.first_exact = first_value
        self.last_exact = last_value
        self.first = jnp.asarray(
            tuple(float(value) for value in first_value), dtype=jnp.float64
        )
        self.last = jnp.asarray(
            tuple(float(value) for value in last_value), dtype=jnp.float64
        )

    @property
    def ambient_dimension(self) -> int:
        return len(self.first_exact)

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return 0.0, 1.0

    def evaluate(self, parameters: Array, /) -> Array:
        parameter = jnp.asarray(parameters, dtype=self.first.dtype)[..., None]
        return (1.0 - parameter) * self.first + parameter * self.last

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        self.is_c1_on(first, last)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("ExactLineCurve derivative bounds support order=1 only.")
        from .._rational import fraction_down, fraction_up

        direction = tuple(
            end - start
            for start, end in zip(self.first_exact, self.last_exact, strict=True)
        )
        return (
            np.asarray(
                tuple(fraction_down(value) for value in direction), dtype=np.float64
            ),
            np.asarray(
                tuple(fraction_up(value) for value in direction), dtype=np.float64
            ),
        )

    @overload
    def evaluate_exact(self, parameter: Fraction, /) -> tuple[Fraction, ...]: ...

    @overload
    def evaluate_exact(
        self, parameter: RadicalElement | float, /
    ) -> tuple[ExactScalar, ...]: ...

    def evaluate_exact(
        self, parameter: ExactScalar | float, /
    ) -> tuple[ExactScalar, ...]:
        value: ExactScalar = (
            parameter
            if isinstance(parameter, (Fraction, RadicalElement))
            else Fraction.from_float(float(parameter))
        )
        return tuple(
            first + value * (last - first)
            for first, last in zip(self.first_exact, self.last_exact, strict=True)
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._rational import fraction_down, fraction_up

        if (
            not np.isfinite(first)
            or not np.isfinite(last)
            or first > last
            or first < 0.0
            or last > 1.0
        ):
            raise ValueError("ExactLineCurve bounds must lie in [0, 1].")
        points = tuple(
            self.evaluate_exact(Fraction.from_float(float(parameter)))
            for parameter in (first, last)
        )
        return np.asarray(
            (
                tuple(
                    fraction_down(min(left, right))
                    for left, right in zip(points[0], points[1], strict=True)
                ),
                tuple(
                    fraction_up(max(left, right))
                    for left, right in zip(points[0], points[1], strict=True)
                ),
            ),
            dtype=np.float64,
        )

    def identity_payload(self) -> dict[str, object]:
        def encode(point: tuple[Fraction, ...], /) -> tuple[tuple[int, int], ...]:
            return tuple((value.numerator, value.denominator) for value in point)

        return {
            "kind": "exact-line-curve",
            "first": encode(self.first_exact),
            "last": encode(self.last_exact),
        }


def _conic_curve_bounds(
    center: tuple[ExactScalar, ...],
    cosine: tuple[ExactScalar, ...],
    sine: tuple[ExactScalar, ...],
    first: float,
    last: float,
    /,
) -> np.ndarray:
    from .._interval_enclosure import interval_cos, interval_sin
    from .._rational import fraction_down, fraction_up

    if not np.isfinite(first) or not np.isfinite(last) or first > last:
        raise ValueError("Conic curve bounds must be finite and ordered.")
    angle = (np.asarray(first, dtype=np.float64), np.asarray(last, dtype=np.float64))
    cosine_lower, cosine_upper = interval_cos(angle)
    sine_lower, sine_upper = interval_sin(angle)
    cosine_range = (
        Fraction.from_float(float(cosine_lower)),
        Fraction.from_float(float(cosine_upper)),
    )
    sine_range = (
        Fraction.from_float(float(sine_lower)),
        Fraction.from_float(float(sine_upper)),
    )

    def multiply(
        first_range: tuple[Fraction, Fraction], second_range: tuple[Fraction, Fraction], /
    ) -> tuple[Fraction, Fraction]:
        products = tuple(a * b for a in first_range for b in second_range)
        return min(products), max(products)

    lower: list[float] = []
    upper: list[float] = []
    for center_value, cosine_value, sine_value in zip(center, cosine, sine, strict=True):
        center_range = coefficient_interval(center_value)
        cosine_coefficient = coefficient_interval(cosine_value)
        sine_coefficient = coefficient_interval(sine_value)
        cosine_term = multiply(cosine_coefficient, cosine_range)
        sine_term = multiply(sine_coefficient, sine_range)
        low = center_range[0] + cosine_term[0] + sine_term[0]
        high = center_range[1] + cosine_term[1] + sine_term[1]
        lower.append(fraction_down(low))
        upper.append(fraction_up(high))
    return np.stack((np.asarray(lower), np.asarray(upper)))


class CircleCurve(AbstractCurve):
    """Circle with exact source coefficients and a JAX numeric realization."""

    center: Array
    first_axis: Array
    second_axis: Array
    radius: Array
    center_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    first_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    second_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    radius_exact: ExactScalar = eqx.field(static=True)

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        radius: ConvertibleToArray,
        *,
        center_exact: tuple[ExactScalar, ...] | None = None,
        first_axis_exact: tuple[ExactScalar, ...] | None = None,
        second_axis_exact: tuple[ExactScalar, ...] | None = None,
        radius_exact: ExactScalar | None = None,
    ) -> None:
        center_ = _curve_vector(center, "center")
        first, second = _orthonormal_pair(
            _curve_vector(first_axis, "first_axis"),
            _curve_vector(second_axis, "second_axis"),
            "Circle",
        )
        radius_ = np.asarray(radius, dtype=np.float64)
        if radius_.shape != () or not np.isfinite(radius_) or radius_ <= 0.0:
            raise ValueError("CircleCurve.radius must be finite and positive.")
        if center_.shape != first.shape:
            raise ValueError("CircleCurve axes must match the center dimension.")

        def exact_vector(
            values: tuple[ExactScalar, ...] | None, shadow: np.ndarray, name: str, /
        ) -> tuple[ExactScalar, ...]:
            result = (
                tuple(Fraction.from_float(float(value)) for value in shadow)
                if values is None
                else tuple(values)
            )
            if len(result) != shadow.size or any(
                not isinstance(value, (Fraction, RadicalElement)) for value in result
            ):
                raise TypeError(f"CircleCurve.{name} must contain exact source scalars.")
            if not np.array_equal(
                np.asarray(tuple(float(value) for value in result), dtype=np.float64),
                shadow,
            ):
                raise ValueError(
                    f"CircleCurve.{name} shadow must round its exact source."
                )
            return result

        center_exact_ = exact_vector(center_exact, center_, "center_exact")
        first_exact_ = exact_vector(first_axis_exact, first, "first_axis_exact")
        second_exact_ = exact_vector(second_axis_exact, second, "second_axis_exact")
        exact_radius = (
            Fraction.from_float(float(radius_)) if radius_exact is None else radius_exact
        )
        if not isinstance(exact_radius, (Fraction, RadicalElement)) or float(
            exact_radius
        ) != float(radius_):
            raise ValueError("CircleCurve radius shadow must round its exact source.")
        self.center = jnp.asarray(center_, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second, dtype=jnp.float64)
        self.radius = jnp.asarray(radius_, dtype=jnp.float64)
        self.center_exact = center_exact_
        self.first_axis_exact = first_exact_
        self.second_axis_exact = second_exact_
        self.radius_exact = exact_radius

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "exact-circle-curve",
            "center": tuple(coefficient_payload(value) for value in self.center_exact),
            "first_axis": tuple(
                coefficient_payload(value) for value in self.first_axis_exact
            ),
            "second_axis": tuple(
                coefficient_payload(value) for value in self.second_axis_exact
            ),
            "radius": coefficient_payload(self.radius_exact),
        }

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    @property
    def period(self) -> float:
        return tau

    def native_period_symbol(self, /) -> str:
        return "two_pi"

    def evaluate(self, parameters: Array, /) -> Array:
        angle = jnp.asarray(parameters, dtype=self.center.dtype)[..., None]
        return self.center + self.radius * (
            jnp.cos(angle) * self.first_axis + jnp.sin(angle) * self.second_axis
        )

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        self.is_c1_on(first, last)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("CircleCurve derivative bounds support order=1 only.")
        from ...discretization._exact_algebraic_geometry import coefficient_interval
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
        )
        from .._rational import fraction_down, fraction_up

        cosine = interval_cos((np.asarray(first), np.asarray(last)))
        sine = interval_sin((np.asarray(first), np.asarray(last)))
        lower = np.empty((self.ambient_dimension,), dtype=np.float64)
        upper = np.empty_like(lower)
        for axis in range(self.ambient_dimension):
            sine_coefficient = coefficient_interval(
                -self.radius_exact * self.first_axis_exact[axis]
            )
            cosine_coefficient = coefficient_interval(
                self.radius_exact * self.second_axis_exact[axis]
            )
            sine_term = interval_multiply(
                (
                    np.asarray(fraction_down(sine_coefficient[0])),
                    np.asarray(fraction_up(sine_coefficient[1])),
                ),
                sine,
            )
            cosine_term = interval_multiply(
                (
                    np.asarray(fraction_down(cosine_coefficient[0])),
                    np.asarray(fraction_up(cosine_coefficient[1])),
                ),
                cosine,
            )
            derivative = interval_add(sine_term, cosine_term)
            lower[axis], upper[axis] = float(derivative[0]), float(derivative[1])
        return lower, upper

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        cosine = tuple(self.radius_exact * value for value in self.first_axis_exact)
        sine = tuple(self.radius_exact * value for value in self.second_axis_exact)
        return _conic_curve_bounds(self.center_exact, cosine, sine, first, last)


class EllipseCurve(AbstractCurve):
    """Ellipse with exact source coefficients and a JAX numeric realization."""

    center: Array
    first_axis: Array
    second_axis: Array
    first_radius: Array
    second_radius: Array
    center_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    first_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    second_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    first_radius_exact: ExactScalar = eqx.field(static=True)
    second_radius_exact: ExactScalar = eqx.field(static=True)

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        first_radius: ConvertibleToArray,
        second_radius: ConvertibleToArray,
        *,
        center_exact: tuple[ExactScalar, ...] | None = None,
        first_axis_exact: tuple[ExactScalar, ...] | None = None,
        second_axis_exact: tuple[ExactScalar, ...] | None = None,
        first_radius_exact: ExactScalar | None = None,
        second_radius_exact: ExactScalar | None = None,
    ) -> None:
        center_ = _curve_vector(center, "center")
        first, second = _orthonormal_pair(
            _curve_vector(first_axis, "first_axis"),
            _curve_vector(second_axis, "second_axis"),
            "Ellipse",
        )
        radii = tuple(
            np.asarray(value, dtype=np.float64) for value in (first_radius, second_radius)
        )
        if any(
            value.shape != () or not np.isfinite(value) or value <= 0.0 for value in radii
        ):
            raise ValueError("EllipseCurve radii must be finite and positive.")
        if center_.shape != first.shape:
            raise ValueError("EllipseCurve axes must match the center dimension.")

        def exact_vector(
            values: tuple[ExactScalar, ...] | None, shadow: np.ndarray, name: str, /
        ) -> tuple[ExactScalar, ...]:
            result = (
                tuple(Fraction.from_float(float(value)) for value in shadow)
                if values is None
                else tuple(values)
            )
            if len(result) != shadow.size or any(
                not isinstance(value, (Fraction, RadicalElement)) for value in result
            ):
                raise TypeError(f"EllipseCurve.{name} must contain exact source scalars.")
            if not np.array_equal(
                np.asarray(tuple(float(value) for value in result), dtype=np.float64),
                shadow,
            ):
                raise ValueError(
                    f"EllipseCurve.{name} shadow must round its exact source."
                )
            return result

        self.center = jnp.asarray(center_, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second, dtype=jnp.float64)
        self.first_radius = jnp.asarray(radii[0], dtype=jnp.float64)
        self.second_radius = jnp.asarray(radii[1], dtype=jnp.float64)
        self.center_exact = exact_vector(center_exact, center_, "center_exact")
        self.first_axis_exact = exact_vector(first_axis_exact, first, "first_axis_exact")
        self.second_axis_exact = exact_vector(
            second_axis_exact, second, "second_axis_exact"
        )
        self.first_radius_exact = (
            Fraction.from_float(float(radii[0]))
            if first_radius_exact is None
            else first_radius_exact
        )
        self.second_radius_exact = (
            Fraction.from_float(float(radii[1]))
            if second_radius_exact is None
            else second_radius_exact
        )
        if any(
            not isinstance(value, (Fraction, RadicalElement))
            or float(value) != float(shadow)
            for value, shadow in zip(
                (self.first_radius_exact, self.second_radius_exact), radii, strict=True
            )
        ):
            raise ValueError(
                "EllipseCurve radius shadows must round their exact sources."
            )

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "exact-ellipse-curve",
            "center": tuple(coefficient_payload(value) for value in self.center_exact),
            "first_axis": tuple(
                coefficient_payload(value) for value in self.first_axis_exact
            ),
            "second_axis": tuple(
                coefficient_payload(value) for value in self.second_axis_exact
            ),
            "first_radius": coefficient_payload(self.first_radius_exact),
            "second_radius": coefficient_payload(self.second_radius_exact),
        }

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    @property
    def period(self) -> float:
        return tau

    def native_period_symbol(self, /) -> str:
        return "two_pi"

    def evaluate(self, parameters: Array, /) -> Array:
        angle = jnp.asarray(parameters, dtype=self.center.dtype)[..., None]
        return (
            self.center
            + self.first_radius * jnp.cos(angle) * self.first_axis
            + self.second_radius * jnp.sin(angle) * self.second_axis
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        cosine = tuple(self.first_radius_exact * value for value in self.first_axis_exact)
        sine = tuple(self.second_radius_exact * value for value in self.second_axis_exact)
        return _conic_curve_bounds(self.center_exact, cosine, sine, first, last)


class ParabolaCurve(AbstractCurve):
    """Vertex-frame parabola ``vertex + t²/(4f) x + t y``."""

    vertex: Array
    first_axis: Array
    second_axis: Array
    focal_length: Array

    def __init__(
        self,
        vertex: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        focal_length: ConvertibleToArray,
    ) -> None:
        vertex_ = _curve_vector(vertex, "vertex")
        first, second = _orthonormal_pair(
            _curve_vector(first_axis, "first_axis"),
            _curve_vector(second_axis, "second_axis"),
            "Parabola",
        )
        focal = np.asarray(focal_length, dtype=np.float64)
        if (
            vertex_.shape != first.shape
            or focal.shape != ()
            or not np.isfinite(focal)
            or focal <= 0.0
        ):
            raise ValueError(
                "ParabolaCurve requires matching frame dimensions and a positive focal length."
            )
        self.vertex = jnp.asarray(vertex_, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second, dtype=jnp.float64)
        self.focal_length = jnp.asarray(focal, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.vertex.shape[0]

    def evaluate(self, parameters: Array, /) -> Array:
        parameter = jnp.asarray(parameters, dtype=self.vertex.dtype)[..., None]
        return (
            self.vertex
            + parameter**2 / (4.0 * self.focal_length) * self.first_axis
            + parameter * self.second_axis
        )


class HyperbolaCurve(AbstractCurve):
    """Positive hyperbola branch in an orthonormal conic frame."""

    center: Array
    first_axis: Array
    second_axis: Array
    first_radius: Array
    second_radius: Array

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        first_radius: ConvertibleToArray,
        second_radius: ConvertibleToArray,
    ) -> None:
        center_ = _curve_vector(center, "center")
        first, second = _orthonormal_pair(
            _curve_vector(first_axis, "first_axis"),
            _curve_vector(second_axis, "second_axis"),
            "Hyperbola",
        )
        radii = tuple(
            np.asarray(value, dtype=np.float64) for value in (first_radius, second_radius)
        )
        if center_.shape != first.shape or any(
            value.shape != () or not np.isfinite(value) or value <= 0.0 for value in radii
        ):
            raise ValueError(
                "HyperbolaCurve requires matching frame dimensions and positive radii."
            )
        self.center = jnp.asarray(center_, dtype=jnp.float64)
        self.first_axis = jnp.asarray(first, dtype=jnp.float64)
        self.second_axis = jnp.asarray(second, dtype=jnp.float64)
        self.first_radius = jnp.asarray(radii[0], dtype=jnp.float64)
        self.second_radius = jnp.asarray(radii[1], dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    def evaluate(self, parameters: Array, /) -> Array:
        parameter = jnp.asarray(parameters, dtype=self.center.dtype)[..., None]
        return (
            self.center
            + self.first_radius * jnp.cosh(parameter) * self.first_axis
            + self.second_radius * jnp.sinh(parameter) * self.second_axis
        )


def _offset_base_is_c2(base: AbstractCurve, first: float, last: float, /) -> bool:
    if not base.is_c1_on(first, last):
        return False
    if type(base) in (
        LineCurve,
        ExactLineCurve,
        CircleCurve,
        EllipseCurve,
        ParabolaCurve,
        HyperbolaCurve,
    ):
        return True
    if isinstance(base, BSplineCurve):
        lower, upper = Fraction(first), Fraction(last)
        return all(
            base.degree - multiplicity >= 2
            for knot, multiplicity in Counter(base.exact_knots).items()
            if lower < knot < upper
        )
    return False


class _OffsetDirectionDim(Dim, minimum=3):
    """Components of an authored space-curve offset direction."""


@final
class OffsetCurve(AbstractCurve):
    """Normal offset retaining the base parameterization and source structure.

    Positive planar distance uses the right normal. A space curve uses its
    tangent crossed with the authored direction. The offset is defined only
    where this normal is nonzero; continuous queries certify that premise.
    """

    __strict_contract__ = True

    base: AbstractCurve
    distance: Float64[Scalar]
    distance_exact: Fraction = eqx.field(static=True)
    direction: Float64[_OffsetDirectionDim] | None

    def __init__(
        self,
        base: AbstractCurve,
        distance: float | Fraction,
        direction: ConvertibleToArray | None = None,
    ) -> None:
        if not isinstance(base, AbstractCurve):
            raise TypeError("OffsetCurve.base must be an AbstractCurve.")
        numeric = finite_real_scalar(distance, "OffsetCurve.distance")
        exact = distance if isinstance(distance, Fraction) else Fraction(numeric)
        if base.ambient_dimension == 2:
            if direction is not None:
                raise ValueError("A planar offset has an intrinsic right normal.")
            direction_value = None
        elif base.ambient_dimension == 3:
            if direction is None:
                raise ValueError("A space-curve offset requires an authored direction.")
            direction_value = _curve_vector(direction, "OffsetCurve.direction")
            if direction_value.shape != (3,) or (
                abs(np.linalg.norm(direction_value) - 1.0)
                > 128.0 * np.finfo(np.float64).eps
            ):
                raise ValueError("OffsetCurve.direction must be a unit three-vector.")
        else:
            raise ValueError("OffsetCurve requires a planar or space curve.")
        self.base = base
        self.distance = jnp.asarray(numeric, dtype=jnp.float64)
        self.distance_exact = exact
        self.direction = (
            None
            if direction_value is None
            else jnp.asarray(direction_value, dtype=jnp.float64)
        )

    def _parameter_source(self) -> AbstractCurve:
        source = self.base
        while isinstance(source, OffsetCurve):
            source = source.base
        return source

    @property
    def ambient_dimension(self) -> int:
        return self._parameter_source().ambient_dimension

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self._parameter_source().parameter_domain

    @property
    def period(self) -> float | None:
        return self._parameter_source().period

    def native_period_symbol(self, /) -> str | None:
        return self._parameter_source().native_period_symbol()

    def validate_source_bank(self, /) -> None:
        if isinstance(self.distance, Tracer):
            return
        if not np.array_equal(np.asarray(self.distance), float(self.distance_exact)):
            raise ValueError(
                "Concrete offset-distance edits require a newly authored source curve."
            )

    def _evaluate_one(self, parameter: Array, /) -> Array:
        if type(self.base) is OffsetCurve:
            layers: list[OffsetCurve] = []
            base: AbstractCurve = self
            while type(base) is OffsetCurve:
                layers.append(base)
                base = base.base
            if type(base) in (LineCurve, ExactLineCurve):
                point, tangent = jax.jvp(
                    base.evaluate, (parameter,), (jnp.ones_like(parameter),)
                )
                ordered = tuple(reversed(layers))
                distances = jnp.stack(tuple(layer.distance for layer in ordered))
                if base.ambient_dimension == 2:
                    normal = jnp.stack((tangent[1], -tangent[0]))
                    normal = normal / jnp.linalg.norm(normal)

                    def planar_offset(
                        value: Array, distance: Array
                    ) -> tuple[Array, None]:
                        return value + distance * normal, None

                    return jax.lax.scan(planar_offset, point, distances)[0]
                directions: list[Array] = []
                for layer in ordered:
                    if layer.direction is None:
                        raise ValueError("A space-curve offset requires a direction.")
                    directions.append(layer.direction)

                def spatial_offset(
                    value: Array, data: tuple[Array, Array]
                ) -> tuple[Array, None]:
                    distance, direction = data
                    normal = jnp.cross(tangent, direction)
                    return value + distance * normal / jnp.linalg.norm(normal), None

                return jax.lax.scan(
                    spatial_offset, point, (distances, jnp.stack(directions))
                )[0]
        point, tangent = jax.jvp(
            self.base.evaluate, (parameter,), (jnp.ones_like(parameter),)
        )
        normal = (
            jnp.stack((tangent[1], -tangent[0]))
            if self.direction is None
            else jnp.cross(tangent, self.direction)
        )
        return point + self.distance * normal / jnp.linalg.norm(normal)

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim == 0:
            return self._evaluate_one(values)
        return jax.vmap(self._evaluate_one)(values.reshape((-1,))).reshape(
            (*values.shape, self.ambient_dimension)
        )

    def is_c1_on(self, first: float, last: float, /) -> bool:
        from .._interval_enclosure import (
            interval_multiply,
            interval_subtract,
            prepare_interval_function,
        )
        from .._rational import fraction_down, fraction_up

        AbstractCurve.is_c1_on(self, first, last)
        self.validate_source_bank()
        if type(self.base) is OffsetCurve:
            from ._source_correspondence import _affine_curve_coefficients

            if (
                _affine_curve_coefficients(self, (Fraction(0),) * self.ambient_dimension)
                is not None
            ):
                return True
        if not _offset_base_is_c2(self.base, first, last):
            return False
        if self.direction is None and type(self.base) in (
            LineCurve,
            ExactLineCurve,
            CircleCurve,
            EllipseCurve,
            ParabolaCurve,
            HyperbolaCurve,
        ):
            return True
        base = self.base
        spans = base.prepare_spans().spans if isinstance(base, BSplineCurve) else ()

        def derivative(parameter: Array, /) -> Array:
            return jax.jvp(
                base.evaluate, (parameter[0],), (jnp.ones_like(parameter[0]),)
            )[1]

        prepared = (
            prepare_interval_function(derivative, 1)
            if isinstance(base, (ParabolaCurve, HyperbolaCurve))
            else None
        )

        def tangent_bounds(
            lower: float, upper: float, /
        ) -> tuple[np.ndarray, np.ndarray]:
            if isinstance(base, (CircleCurve, EllipseCurve)):
                a = (
                    base.radius_exact
                    if isinstance(base, CircleCurve)
                    else base.first_radius_exact
                )
                b = (
                    base.radius_exact
                    if isinstance(base, CircleCurve)
                    else base.second_radius_exact
                )
                bounds = _conic_curve_bounds(
                    (Fraction(0),) * base.ambient_dimension,
                    tuple(b * value for value in base.second_axis_exact),
                    tuple(-a * value for value in base.first_axis_exact),
                    lower,
                    upper,
                )
                return bounds[0], bounds[1]
            if isinstance(base, BSplineCurve):
                intervals = tuple(
                    span.enclose(
                        (
                            max(Fraction(lower), span.parameter_bounds[0]),
                            min(Fraction(upper), span.parameter_bounds[1]),
                        )
                    )
                    for span in spans
                    if span.parameter_bounds[0] <= Fraction(upper)
                    and span.parameter_bounds[1] >= Fraction(lower)
                )
                return (
                    np.asarray(
                        [
                            fraction_down(
                                min(
                                    value.first_derivative_bounds[axis][0]
                                    for value in intervals
                                )
                            )
                            for axis in range(base.ambient_dimension)
                        ],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [
                            fraction_up(
                                max(
                                    value.first_derivative_bounds[axis][1]
                                    for value in intervals
                                )
                            )
                            for axis in range(base.ambient_dimension)
                        ],
                        dtype=np.float64,
                    ),
                )
            if prepared is not None:
                low, high = prepared.evaluate(
                    np.asarray(((lower,),), dtype=np.float64),
                    np.asarray(((upper,),), dtype=np.float64),
                )
                return low[0], high[0]
            return base.derivative_bounds(lower, upper)

        direction = None if self.direction is None else np.asarray(self.direction)
        pending = [(first, last)]
        while pending:
            lower, upper = pending.pop()
            low, high = tangent_bounds(lower, upper)
            if direction is None:
                normal_low = np.asarray((low[1], -high[0]), dtype=np.float64)
                normal_high = np.asarray((high[1], -low[0]), dtype=np.float64)
            else:
                left = np.asarray((1, 2, 0), dtype=np.intp)
                right = np.asarray((2, 0, 1), dtype=np.intp)
                normal_low, normal_high = interval_subtract(
                    interval_multiply(
                        (low[left], high[left]), (direction[right], direction[right])
                    ),
                    interval_multiply(
                        (low[right], high[right]), (direction[left], direction[left])
                    ),
                )
            normal_low, normal_high = np.asarray(normal_low), np.asarray(normal_high)
            if not np.all(np.isfinite(normal_low)) or not np.all(
                np.isfinite(normal_high)
            ):
                return False
            if np.any(normal_low > 0.0) or np.any(normal_high < 0.0):
                continue
            middle = lower * 0.5 + upper * 0.5
            if (
                isinstance(base, (LineCurve, ExactLineCurve))
                or not lower < middle < upper
            ):
                return False
            pending.extend(((middle, upper), (lower, middle)))
        return True

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._rational import fraction_down, fraction_up
        from ._source_correspondence import _affine_curve_coefficients

        AbstractCurve.is_c1_on(self, first, last)
        self.validate_source_bank()
        affine = _affine_curve_coefficients(self, (Fraction(0),) * self.ambient_dimension)
        if affine is not None:
            origin, direction = affine
            endpoints = tuple(
                (
                    coefficient_interval(value + Fraction(first) * slope),
                    coefficient_interval(value + Fraction(last) * slope),
                )
                for value, slope in zip(origin, direction, strict=True)
            )
            return np.asarray(
                (
                    tuple(fraction_down(min(a[0], b[0])) for a, b in endpoints),
                    tuple(fraction_up(max(a[1], b[1])) for a, b in endpoints),
                ),
                dtype=np.float64,
            )

        if not self.is_c1_on(first, last):
            raise NotImplementedError(
                "Offset curve normal regularity is not established."
            )
        bounds = self.base.bounding_box(first, last)
        if not np.all(np.isfinite(bounds)):
            raise ValueError("Offset curve source bounds must be finite.")
        distance = abs(self.distance_exact)
        return np.asarray(
            (
                tuple(
                    fraction_down(Fraction(float(value)) - distance)
                    for value in bounds[0]
                ),
                tuple(
                    fraction_up(Fraction(float(value)) + distance) for value in bounds[1]
                ),
            ),
            dtype=np.float64,
        )

    def identity_payload(self, /) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        self.validate_source_bank()
        return {
            "kind": "normal-offset-curve",
            "base": carrier_payload(self.base),
            "distance": coefficient_payload(self.distance_exact),
            "direction": None
            if self.direction is None
            else array_tree_fingerprint(self.direction),
        }


@final
class RuledSurface(AbstractSurfacePatch):
    """Linear interpolation between two space curves with one shared u parameter."""

    first: AbstractCurve
    second: AbstractCurve

    def __init__(self, first: AbstractCurve, second: AbstractCurve) -> None:
        if not isinstance(first, AbstractCurve) or not isinstance(second, AbstractCurve):
            raise TypeError("RuledSurface rails must be AbstractCurve values.")
        if first.ambient_dimension != 3 or second.ambient_dimension != 3:
            raise ValueError("RuledSurface rails must be three-dimensional.")
        domains = tuple(
            value.parameter_domain
            for value in (first, second)
            if value.parameter_domain is not None
        )
        if domains and max(domain[0] for domain in domains) >= min(
            domain[1] for domain in domains
        ):
            raise ValueError(
                "RuledSurface rails must have an overlapping parameter domain."
            )
        self.first, self.second = first, second

    @property
    def parameter_domain(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        domains = tuple(
            value.parameter_domain
            for value in (self.first, self.second)
            if value.parameter_domain is not None
        )
        if not domains:
            return None
        return (
            (max(domain[0] for domain in domains), 0.0),
            (min(domain[1] for domain in domains), 1.0),
        )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        if (
            self.first.period == self.second.period
            and self.first.native_period_symbol() == self.second.native_period_symbol()
        ):
            return self.first.period, None
        return None, None

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return (
            self.first.native_period_symbol() if self.periods[0] is not None else None
        ), None

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        u, v = values[..., 0], values[..., 1, None]
        return (1.0 - v) * self.first.evaluate(u) + v * self.second.evaluate(u)

    def validate_parameter_box(self, parameter_box: ConvertibleToArray, /) -> Array:
        box = super().validate_parameter_box(parameter_box)
        values = np.asarray(box)
        if values[0, 1] < 0.0 or values[1, 1] > 1.0:
            raise ValueError("RuledSurface v parameters must lie in [0, 1].")
        return box

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        box = np.asarray(self.validate_parameter_box(parameter_box))
        first, last = float(box[0, 0]), float(box[1, 0])
        return self.first.is_c1_on(first, last) and self.second.is_c1_on(first, last)

    def is_c1_along(
        self, fixed_axis: int, fixed_value: float, first: float, last: float, /
    ) -> bool:
        super().is_c1_along(fixed_axis, fixed_value, first, last)
        if fixed_axis == 0:
            if not 0.0 <= first <= last <= 1.0:
                raise ValueError("RuledSurface v parameters must lie in [0, 1].")
            return True
        if not 0.0 <= fixed_value <= 1.0:
            raise ValueError("RuledSurface v parameters must lie in [0, 1].")
        return self.first.is_c1_on(first, last) and self.second.is_c1_on(first, last)

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        box = np.asarray(parameter_box, dtype=np.float64)
        if box.shape != (2, 2) or not np.all(np.isfinite(box)) or np.any(box[1] < box[0]):
            raise ValueError("Ruled source interval boxes must be finite and ordered.")
        domain = self.parameter_domain
        if domain is not None and (box[0, 0] < domain[0][0] or box[1, 0] > domain[1][0]):
            raise ValueError("Ruled source interval must lie within its rail domain.")
        first = self.first.bounding_box(float(box[0, 0]), float(box[1, 0]))
        second = self.second.bounding_box(float(box[0, 0]), float(box[1, 0]))
        v_lower, v_upper = float(box[0, 1]), float(box[1, 1])
        convex = 0.0 <= v_lower <= v_upper <= 1.0
        if not convex:
            crosses_lower = v_lower < 0.0 <= v_upper
            crosses_upper = v_lower <= 1.0 < v_upper
            if crosses_lower == crosses_upper:
                raise ValueError(
                    "Ruled source interval may cross only one exact v-domain boundary."
                )
        from .._interval_enclosure import (
            interval_add,
            interval_multiply,
            interval_subtract,
        )

        v_interval = (np.asarray(v_lower), np.asarray(v_upper))
        complement = interval_subtract((np.asarray(1.0), np.asarray(1.0)), v_interval)
        first_part = interval_multiply(complement, (first[0], first[1]))
        second_part = interval_multiply(v_interval, (second[0], second[1]))
        lower, upper = interval_add(first_part, second_part)
        if convex:
            lower = np.maximum(lower, np.minimum(first[0], second[0]))
            upper = np.minimum(upper, np.maximum(first[1], second[1]))
        return np.stack((lower, upper))

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
            interval_subtract,
        )
        from .._rational import fraction_down, fraction_up

        raw_box = np.asarray(parameter_box, dtype=np.float64)
        if (
            raw_box.shape != (2, 2)
            or not np.all(np.isfinite(raw_box))
            or np.any(raw_box[1] <= raw_box[0])
        ):
            raise ValueError(
                "Ruled surface derivative boxes must be finite and increasing."
            )
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Ruled surface derivative bounds support order=1 only.")
        domain = self.parameter_domain
        u_extension = v_extension = False

        def one_boundary_interval(
            lower: float, upper: float, domain_lower: float, domain_upper: float, /
        ) -> tuple[bool, bool]:
            if domain_lower <= lower and upper <= domain_upper:
                return True, False
            crosses_lower = lower < domain_lower <= upper <= domain_upper
            crosses_upper = domain_lower <= lower <= domain_upper < upper
            return crosses_lower or crosses_upper, crosses_lower or crosses_upper

        if domain is not None:
            u_valid, u_extension = one_boundary_interval(
                float(raw_box[0, 0]),
                float(raw_box[1, 0]),
                domain[0][0],
                domain[1][0],
            )
            v_valid, v_extension = one_boundary_interval(
                float(raw_box[0, 1]),
                float(raw_box[1, 1]),
                domain[0][1],
                domain[1][1],
            )
            if not u_valid or not v_valid:
                raise ValueError(
                    "Ruled derivative extension may cross only one exact source boundary per axis."
                )
        analytic_rails = all(
            isinstance(rail, SurfaceMappedCurve)
            and isinstance(rail.surface, CylinderPatch)
            and isinstance(rail.pcurve, ConicRootLinePCurve)
            for rail in (self.first, self.second)
        )
        if u_extension and not analytic_rails:
            raise ValueError(
                "Ruled derivative u extension requires exact analytic cylinder rails."
            )
        extension = u_extension or v_extension
        box = (
            raw_box
            if extension
            else np.asarray(self.validate_parameter_box(raw_box), dtype=np.float64)
        )
        if analytic_rails:
            rail_first = self.first
            rail_second = self.second
            assert isinstance(rail_first, SurfaceMappedCurve)
            assert isinstance(rail_second, SurfaceMappedCurve)
            patch_first, source_first = rail_first.surface, rail_first.pcurve
            patch_second, source_second = rail_second.surface, rail_second.pcurve
            assert isinstance(patch_first, CylinderPatch)
            assert isinstance(patch_second, CylinderPatch)
            assert isinstance(source_first, ConicRootLinePCurve)
            assert isinstance(source_second, ConicRootLinePCurve)

            def exact_interval(value: ExactScalar, /) -> Interval:
                lower, upper = coefficient_interval(value, bits=128)
                return (
                    np.asarray(fraction_down(lower), dtype=np.float64),
                    np.asarray(fraction_up(upper), dtype=np.float64),
                )

            def rail_box(
                patch: CylinderPatch, source: ConicRootLinePCurve, /
            ) -> np.ndarray:
                chart = np.asarray(
                    source.bounding_box(float(box[0, 0]), float(box[1, 0])),
                    dtype=np.float64,
                )
                angle = (chart[0, 0], chart[1, 0])
                height = (chart[0, 1], chart[1, 1])
                cosine = interval_cos((np.asarray(angle[0]), np.asarray(angle[1])))
                sine = interval_sin((np.asarray(angle[0]), np.asarray(angle[1])))
                lower: list[float] = []
                upper: list[float] = []
                for axis in range(3):
                    radial = interval_add(
                        interval_multiply(
                            exact_interval(
                                patch.radius_exact * patch.first_axis_exact[axis]
                            ),
                            cosine,
                        ),
                        interval_multiply(
                            exact_interval(
                                patch.radius_exact * patch.second_axis_exact[axis]
                            ),
                            sine,
                        ),
                    )
                    axial = interval_multiply(
                        exact_interval(patch.axis_exact[axis]),
                        (np.asarray(height[0]), np.asarray(height[1])),
                    )
                    value = interval_add(
                        interval_add(exact_interval(patch.origin_exact[axis]), radial),
                        axial,
                    )
                    lower.append(float(value[0]))
                    upper.append(float(value[1]))
                return np.asarray((lower, upper), dtype=np.float64)

            first_box = rail_box(patch_first, source_first)
            second_box = rail_box(patch_second, source_second)
            v_interval = (np.asarray(box[0, 1]), np.asarray(box[1, 1]))
            complement = interval_subtract((np.asarray(1.0), np.asarray(1.0)), v_interval)
            first_velocity = tuple(
                exact_interval(source_first.height_direction_exact * value)
                for value in patch_first.axis_exact
            )
            second_velocity = tuple(
                exact_interval(source_second.height_direction_exact * value)
                for value in patch_second.axis_exact
            )
            lower = np.empty((3, 2), dtype=np.float64)
            upper = np.empty_like(lower)
            for axis in range(3):
                along = interval_add(
                    interval_multiply(complement, first_velocity[axis]),
                    interval_multiply(v_interval, second_velocity[axis]),
                )
                across = interval_subtract(
                    (second_box[0, axis], second_box[1, axis]),
                    (first_box[0, axis], first_box[1, axis]),
                )
                lower[axis] = float(along[0]), float(across[0])
                upper[axis] = float(along[1]), float(across[1])
            return lower, upper
        if extension:
            first, second = float(box[0, 0]), float(box[1, 0])
            if not self.first.is_c1_on(first, second) or not self.second.is_c1_on(
                first, second
            ):
                raise NotImplementedError(
                    "Ruled source rails are not C1 on the analytic v extension."
                )
        elif not self.is_c1_on(box):
            raise NotImplementedError("Ruled surface rail continuity is not established.")
        from .._interval_enclosure import prepare_interval_function

        prepared = prepare_interval_function(jax.jacfwd(self.evaluate), 2)
        lower, upper = prepared.evaluate(box[0:1], box[1:2])
        return lower[0], upper[0]

    def identity_payload(self, /) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "ruled-surface",
            "first": carrier_payload(self.first),
            "second": carrier_payload(self.second),
        }


class ExtrusionSurface(AbstractSurfacePatch):
    """Linear extrusion ``curve(u) + v * direction`` of a space curve."""

    curve: AbstractCurve
    direction: Array

    def __init__(self, curve: AbstractCurve, direction: ConvertibleToArray) -> None:
        if not isinstance(curve, AbstractCurve) or curve.ambient_dimension != 3:
            raise TypeError("ExtrusionSurface.curve must be a three-dimensional curve.")
        direction_ = _curve_vector(direction, "direction")
        if direction_.shape != (3,) or np.linalg.norm(direction_) == 0.0:
            raise ValueError("ExtrusionSurface.direction must be a nonzero 3D vector.")
        self.curve = curve
        self.direction = jnp.asarray(direction_, dtype=jnp.float64)

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return self.curve.period, None

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return self.curve.native_period_symbol(), None

    @property
    def parameter_domain(
        self, /
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        profile = self.curve.parameter_domain
        if profile is None:
            return None
        return (profile[0], -np.inf), (profile[1], np.inf)

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.direction.dtype)
        return (
            self.curve.evaluate(parameters_[..., 0])
            + parameters_[..., 1:2] * self.direction
        )

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        box = self.validate_parameter_box(parameter_box)
        return self.curve.is_c1_on(float(box[0, 0]), float(box[1, 0]))

    def is_c1_along(
        self, fixed_axis: int, fixed_value: float, first: float, last: float, /
    ) -> bool:
        super().is_c1_along(fixed_axis, fixed_value, first, last)
        return fixed_axis == 0 or self.curve.is_c1_on(first, last)

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enclose ``(curve'(u), direction)``; the generator column is the exact vector."""
        box = np.asarray(self.validate_parameter_box(parameter_box), dtype=np.float64)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("ExtrusionSurface derivative bounds support order=1 only.")
        if not self.curve.is_c1_on(float(box[0, 0]), float(box[1, 0])):
            raise NotImplementedError(
                "ExtrusionSurface profile is not C1 on the requested interval."
            )
        profile_lower, profile_upper = self.curve.derivative_bounds(
            float(box[0, 0]), float(box[1, 0])
        )
        direction = np.asarray(self.direction, dtype=np.float64)
        return (
            np.stack((np.asarray(profile_lower, dtype=np.float64), direction), axis=1),
            np.stack((np.asarray(profile_upper, dtype=np.float64), direction), axis=1),
        )


class RevolutionSurface(AbstractSurfacePatch):
    """Surface of revolution of a space curve around an authored axis."""

    curve: AbstractCurve
    axis_origin: Array
    axis_direction: Array

    def __init__(
        self,
        curve: AbstractCurve,
        axis_origin: ConvertibleToArray,
        axis_direction: ConvertibleToArray,
    ) -> None:
        if not isinstance(curve, AbstractCurve) or curve.ambient_dimension != 3:
            raise TypeError("RevolutionSurface.curve must be a three-dimensional curve.")
        origin = _curve_vector(axis_origin, "axis_origin")
        direction = _curve_vector(axis_direction, "axis_direction")
        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError("RevolutionSurface axis vectors must be three-dimensional.")
        length = float(np.linalg.norm(direction))
        if abs(length - 1.0) > 128.0 * np.finfo(np.float64).eps:
            raise ValueError("RevolutionSurface.axis_direction must be a unit vector.")
        self.curve = curve
        self.axis_origin = jnp.asarray(origin, dtype=jnp.float64)
        self.axis_direction = jnp.asarray(direction / length, dtype=jnp.float64)

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return 2.0 * np.pi, self.curve.period

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return "two_pi", self.curve.native_period_symbol()

    @property
    def parameter_domain(
        self, /
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        profile = self.curve.parameter_domain
        if profile is None:
            return None
        return ((0.0, profile[0]), (2.0 * np.pi, profile[1]))

    def identity_payload(self, /) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "revolution-surface",
            "profile": carrier_payload(self.curve),
            "axis_origin": tuple(
                float(value).hex() for value in np.asarray(self.axis_origin)
            ),
            "axis_direction": tuple(
                float(value).hex() for value in np.asarray(self.axis_direction)
            ),
        }

    def evaluate(self, parameters: Array, /) -> Array:
        # The axial part of the profile point does not rotate. Keeping it out of
        # the angle terms leaves interval enclosures of this map, its Jacobian
        # and Hessian free of a dependency on the angle interval that the
        # equivalent Rodrigues form carries through ``cos`` twice.
        parameters_ = jnp.asarray(parameters, dtype=self.axis_origin.dtype)
        angle = parameters_[..., 0:1]
        relative = self.curve.evaluate(parameters_[..., 1]) - self.axis_origin
        axis = self.axis_direction
        axial = axis * jnp.sum(relative * axis, axis=-1, keepdims=True)
        radial = relative - axial
        return (
            self.axis_origin
            + axial
            + radial * jnp.cos(angle)
            + jnp.cross(axis, radial) * jnp.sin(angle)
        )

    def _validate_source_parameter_box(
        self, parameter_box: ConvertibleToArray, /
    ) -> np.ndarray:
        box = np.asarray(parameter_box, dtype=np.float64)
        if box.shape != (2, 2) or not np.all(np.isfinite(box)) or np.any(box[1] < box[0]):
            raise ValueError(
                "A surface parameter box must be finite, shaped (2, 2), and ordered."
            )
        domain = self.parameter_domain
        if domain is not None and (box[0, 1] < domain[0][1] or box[1, 1] > domain[1][1]):
            raise ValueError(
                "The profile parameter box must lie within its source domain."
            )
        for axis, period in enumerate(self.periods):
            if period is not None and box[1, axis] - box[0, axis] > period * (
                1.0 + 128.0 * np.finfo(np.float64).eps
            ):
                raise ValueError(
                    "A periodic parameter interval may cover at most one period."
                )
        return box

    def bounding_box(self, parameter_box: ConvertibleToArray, /) -> np.ndarray:
        from ...discretization._exact_algebraic_geometry import coefficient_interval
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
            interval_subtract,
        )
        from .._rational import fraction_down, fraction_up, rational_vector

        if isinstance(self.curve, CircleCurve):
            box = self._validate_source_parameter_box(parameter_box)
            axis_origin = rational_vector(np.asarray(self.axis_origin, dtype=np.float64))
            axis = rational_vector(np.asarray(self.axis_direction, dtype=np.float64))
            profile = self.curve

            def exact_interval(value: ExactScalar, /) -> Interval:
                lower, upper = coefficient_interval(value, bits=128)
                return (
                    np.asarray(fraction_down(lower), dtype=np.float64),
                    np.asarray(fraction_up(upper), dtype=np.float64),
                )

            profile_relative_center = tuple(
                profile.center_exact[index] - axis_origin[index] for index in range(3)
            )
            profile_cosine_exact = tuple(
                profile.radius_exact * profile.first_axis_exact[index]
                for index in range(3)
            )
            profile_sine_exact = tuple(
                profile.radius_exact * profile.second_axis_exact[index]
                for index in range(3)
            )
            profile_angle = (
                np.asarray(box[0, 1], dtype=np.float64),
                np.asarray(box[1, 1], dtype=np.float64),
            )
            profile_cosine = interval_cos(profile_angle)
            profile_sine = interval_sin(profile_angle)
            relative = tuple(
                interval_add(
                    interval_add(
                        exact_interval(profile_relative_center[index]),
                        interval_multiply(
                            exact_interval(profile_cosine_exact[index]),
                            profile_cosine,
                        ),
                    ),
                    interval_multiply(
                        exact_interval(profile_sine_exact[index]), profile_sine
                    ),
                )
                for index in range(3)
            )
            axis_intervals = tuple(exact_interval(value) for value in axis)
            axial = (np.asarray(0.0), np.asarray(0.0))
            for component in range(3):
                axial = interval_add(
                    axial,
                    interval_multiply(axis_intervals[component], relative[component]),
                )
            cross_axis = (
                interval_subtract(
                    interval_multiply(axis_intervals[1], relative[2]),
                    interval_multiply(axis_intervals[2], relative[1]),
                ),
                interval_subtract(
                    interval_multiply(axis_intervals[2], relative[0]),
                    interval_multiply(axis_intervals[0], relative[2]),
                ),
                interval_subtract(
                    interval_multiply(axis_intervals[0], relative[1]),
                    interval_multiply(axis_intervals[1], relative[0]),
                ),
            )
            angle = (
                np.asarray(box[0, 0], dtype=np.float64),
                np.asarray(box[1, 0], dtype=np.float64),
            )
            cosine, sine = interval_cos(angle), interval_sin(angle)
            one_minus_cosine = interval_subtract(
                (np.asarray(1.0), np.asarray(1.0)), cosine
            )
            lower, upper = [], []
            for component in range(3):
                rotated = interval_add(
                    interval_add(
                        interval_multiply(relative[component], cosine),
                        interval_multiply(cross_axis[component], sine),
                    ),
                    interval_multiply(
                        interval_multiply(axis_intervals[component], axial),
                        one_minus_cosine,
                    ),
                )
                point = interval_add(exact_interval(axis_origin[component]), rotated)
                lower.append(float(point[0]))
                upper.append(float(point[1]))
            return np.asarray((lower, upper), dtype=np.float64)

        if not isinstance(self.curve, (LineCurve, ExactLineCurve)):
            return super().bounding_box(parameter_box)
        box = self._validate_source_parameter_box(parameter_box)
        if isinstance(self.curve, LineCurve):
            profile_origin = self.curve.origin_exact
            profile_direction = self.curve.direction_exact
        else:
            profile_origin = self.curve.first_exact
            profile_direction = tuple(
                last - first
                for first, last in zip(
                    self.curve.first_exact, self.curve.last_exact, strict=True
                )
            )
        axis_origin = rational_vector(np.asarray(self.axis_origin, dtype=np.float64))
        axis = rational_vector(np.asarray(self.axis_direction, dtype=np.float64))

        def dot(
            first: tuple[ExactScalar, ...],
            second: tuple[ExactScalar, ...],
            /,
        ) -> ExactScalar:
            return sum(
                (left * right for left, right in zip(first, second, strict=True)),
                Fraction(0),
            )

        def exact_interval(value: ExactScalar, /) -> Interval:
            lower, upper = coefficient_interval(value, bits=128)
            return (
                np.asarray(fraction_down(lower), dtype=np.float64),
                np.asarray(fraction_up(upper), dtype=np.float64),
            )

        relative_origin = tuple(
            profile_origin[index] - axis_origin[index] for index in range(3)
        )
        axial_origin = dot(axis, relative_origin)
        axial_direction = dot(axis, profile_direction)
        radial_origin = tuple(
            relative_origin[index] - axis[index] * axial_origin for index in range(3)
        )
        radial_direction = tuple(
            profile_direction[index] - axis[index] * axial_direction for index in range(3)
        )
        v_lower = Fraction.from_float(float(box[0, 1]))
        v_upper = Fraction.from_float(float(box[1, 1]))
        axial_values = (
            axial_origin + axial_direction * v_lower,
            axial_origin + axial_direction * v_upper,
        )
        axial = (
            exact_interval(min(axial_values))[0],
            exact_interval(max(axial_values))[1],
        )
        radial: list[Interval] = []
        for origin, direction in zip(radial_origin, radial_direction, strict=True):
            endpoints = (origin + direction * v_lower, origin + direction * v_upper)
            radial.append(
                (
                    exact_interval(min(endpoints))[0],
                    exact_interval(max(endpoints))[1],
                )
            )
        axis_intervals = tuple(exact_interval(value) for value in axis)
        cross_axis = (
            interval_subtract(
                interval_multiply(axis_intervals[1], radial[2]),
                interval_multiply(axis_intervals[2], radial[1]),
            ),
            interval_subtract(
                interval_multiply(axis_intervals[2], radial[0]),
                interval_multiply(axis_intervals[0], radial[2]),
            ),
            interval_subtract(
                interval_multiply(axis_intervals[0], radial[1]),
                interval_multiply(axis_intervals[1], radial[0]),
            ),
        )
        angle = (np.asarray(box[0, 0]), np.asarray(box[1, 0]))
        cosine = interval_cos(angle)
        sine = interval_sin(angle)
        lower: list[float] = []
        upper: list[float] = []
        for component in range(3):
            rotated = interval_add(
                interval_add(
                    interval_multiply(axis_intervals[component], axial),
                    interval_multiply(radial[component], cosine),
                ),
                interval_multiply(cross_axis[component], sine),
            )
            point = interval_add(exact_interval(axis_origin[component]), rotated)
            lower.append(float(point[0]))
            upper.append(float(point[1]))
        return np.asarray((lower, upper), dtype=np.float64)

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        box = self._validate_source_parameter_box(parameter_box)
        return self.curve.is_c1_on(float(box[0, 1]), float(box[1, 1]))

    def is_c1_along(
        self, fixed_axis: int, fixed_value: float, first: float, last: float, /
    ) -> bool:
        super().is_c1_along(fixed_axis, fixed_value, first, last)
        return fixed_axis == 1 or self.curve.is_c1_on(first, last)

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        from ...discretization._exact_algebraic_geometry import coefficient_interval
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
            interval_subtract,
        )
        from .._rational import fraction_down, fraction_up, rational_vector

        box = np.asarray(
            self._validate_source_parameter_box(parameter_box), dtype=np.float64
        )
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("RevolutionSurface derivative bounds support order=1 only.")
        profile = self.curve
        if not isinstance(profile, (CircleCurve, LineCurve, ExactLineCurve)):
            raise NotImplementedError(
                "RevolutionSurface derivative bounds require a source CircleCurve or line profile."
            )
        if not profile.is_c1_on(float(box[0, 1]), float(box[1, 1])):
            raise NotImplementedError(
                "RevolutionSurface profile is not C1 on the requested interval."
            )

        def exact_dot(
            first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
        ) -> ExactScalar:
            return sum(
                (left * right for left, right in zip(first, second, strict=True)),
                Fraction(0),
            )

        def exact_interval(value: ExactScalar, /) -> tuple[np.ndarray, np.ndarray]:
            lower, upper = coefficient_interval(value, bits=128)
            return np.asarray(fraction_down(lower)), np.asarray(fraction_up(upper))

        def array_interval(value: Interval, /) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.asarray(value[0], dtype=np.float64),
                np.asarray(value[1], dtype=np.float64),
            )

        def fixed(value: float, /) -> tuple[np.ndarray, np.ndarray]:
            scalar = np.asarray(value, dtype=np.float64)
            return scalar, scalar

        axis = np.asarray(self.axis_direction, dtype=np.float64)
        axis_exact = rational_vector(axis)
        origin_exact = rational_vector(np.asarray(self.axis_origin))
        if isinstance(profile, CircleCurve):
            relative_center = tuple(
                profile.center_exact[index] - origin_exact[index] for index in range(3)
            )
            cosine_coefficients = tuple(
                profile.radius_exact * profile.first_axis_exact[index]
                for index in range(3)
            )
            sine_coefficients = tuple(
                profile.radius_exact * profile.second_axis_exact[index]
                for index in range(3)
            )
            profile_angle = (np.asarray(box[0, 1]), np.asarray(box[1, 1]))
            profile_cosine = array_interval(interval_cos(profile_angle))
            profile_sine = array_interval(interval_sin(profile_angle))
            relative = tuple(
                interval_add(
                    interval_add(
                        exact_interval(relative_center[index]),
                        interval_multiply(
                            exact_interval(cosine_coefficients[index]),
                            profile_cosine,
                        ),
                    ),
                    interval_multiply(
                        exact_interval(sine_coefficients[index]), profile_sine
                    ),
                )
                for index in range(3)
            )
            tangent = tuple(
                interval_add(
                    interval_multiply(
                        exact_interval(-cosine_coefficients[index]), profile_sine
                    ),
                    interval_multiply(
                        exact_interval(sine_coefficients[index]), profile_cosine
                    ),
                )
                for index in range(3)
            )
            axis_intervals = tuple(exact_interval(value) for value in axis_exact)

            def cross_interval(vector: tuple[Interval, ...], /) -> tuple[Interval, ...]:
                return (
                    interval_subtract(
                        interval_multiply(axis_intervals[1], vector[2]),
                        interval_multiply(axis_intervals[2], vector[1]),
                    ),
                    interval_subtract(
                        interval_multiply(axis_intervals[2], vector[0]),
                        interval_multiply(axis_intervals[0], vector[2]),
                    ),
                    interval_subtract(
                        interval_multiply(axis_intervals[0], vector[1]),
                        interval_multiply(axis_intervals[1], vector[0]),
                    ),
                )

            axial = (np.asarray(0.0), np.asarray(0.0))
            axial_tangent = (np.asarray(0.0), np.asarray(0.0))
            for component in range(3):
                axial = interval_add(
                    axial,
                    interval_multiply(axis_intervals[component], relative[component]),
                )
                axial_tangent = interval_add(
                    axial_tangent,
                    interval_multiply(axis_intervals[component], tangent[component]),
                )
            cross_position = cross_interval(relative)
            cross_tangent = cross_interval(tangent)
            angle = (np.asarray(box[0, 0]), np.asarray(box[1, 0]))
            angle_cosine = array_interval(interval_cos(angle))
            angle_sine = array_interval(interval_sin(angle))
            negative_angle_sine = (-angle_sine[1], -angle_sine[0])
            one_minus_cosine = interval_subtract(
                (np.asarray(1.0), np.asarray(1.0)), angle_cosine
            )
            first_column: list[tuple[np.ndarray, np.ndarray]] = []
            second_column: list[tuple[np.ndarray, np.ndarray]] = []
            for component in range(3):
                first_column.append(
                    array_interval(
                        interval_add(
                            interval_add(
                                interval_multiply(
                                    relative[component], negative_angle_sine
                                ),
                                interval_multiply(
                                    cross_position[component], angle_cosine
                                ),
                            ),
                            interval_multiply(
                                interval_multiply(axis_intervals[component], axial),
                                angle_sine,
                            ),
                        )
                    )
                )
                second_column.append(
                    array_interval(
                        interval_add(
                            interval_add(
                                interval_multiply(tangent[component], angle_cosine),
                                interval_multiply(cross_tangent[component], angle_sine),
                            ),
                            interval_multiply(
                                interval_multiply(
                                    axis_intervals[component], axial_tangent
                                ),
                                one_minus_cosine,
                            ),
                        )
                    )
                )
            lower = np.column_stack(
                (
                    tuple(float(value[0]) for value in first_column),
                    tuple(float(value[0]) for value in second_column),
                )
            )
            upper = np.column_stack(
                (
                    tuple(float(value[1]) for value in first_column),
                    tuple(float(value[1]) for value in second_column),
                )
            )
            return lower, upper
        else:
            if isinstance(profile, LineCurve):
                profile_origin_exact = profile.origin_exact
                profile_direction_exact = profile.direction_exact
            else:
                profile_origin_exact = profile.first_exact
                profile_direction_exact = tuple(
                    last - first
                    for first, last in zip(
                        profile.first_exact, profile.last_exact, strict=True
                    )
                )
            relative_origin = tuple(
                profile_origin_exact[index] - origin_exact[index] for index in range(3)
            )
            axial_origin = exact_dot(axis_exact, relative_origin)
            axial_direction = exact_dot(axis_exact, profile_direction_exact)
            radial_origin = tuple(
                relative_origin[index] - axis_exact[index] * axial_origin
                for index in range(3)
            )
            radial_direction = tuple(
                profile_direction_exact[index] - axis_exact[index] * axial_direction
                for index in range(3)
            )
            first_parameter = Fraction.from_float(float(box[0, 1]))
            last_parameter = Fraction.from_float(float(box[1, 1]))
            radial_lower = np.empty((3,), dtype=np.float64)
            radial_upper = np.empty((3,), dtype=np.float64)
            tangent_lower = np.empty((3,), dtype=np.float64)
            tangent_upper = np.empty((3,), dtype=np.float64)
            for component in range(3):
                first_value = (
                    radial_origin[component]
                    + first_parameter * radial_direction[component]
                )
                last_value = (
                    radial_origin[component]
                    + last_parameter * radial_direction[component]
                )
                low = first_value if first_value <= last_value else last_value
                high = last_value if first_value <= last_value else first_value
                radial_lower[component] = exact_interval(low)[0]
                radial_upper[component] = exact_interval(high)[1]
                tangent_lower[component], tangent_upper[component] = exact_interval(
                    radial_direction[component]
                )
            axial_tangent = exact_interval(axial_direction)

        def cross_axis(
            vector_lower: np.ndarray, vector_upper: np.ndarray, /
        ) -> tuple[np.ndarray, np.ndarray]:
            vector = tuple(
                (np.asarray(vector_lower[index]), np.asarray(vector_upper[index]))
                for index in range(3)
            )
            components = (
                interval_subtract(
                    interval_multiply(fixed(float(axis[1])), vector[2]),
                    interval_multiply(fixed(float(axis[2])), vector[1]),
                ),
                interval_subtract(
                    interval_multiply(fixed(float(axis[2])), vector[0]),
                    interval_multiply(fixed(float(axis[0])), vector[2]),
                ),
                interval_subtract(
                    interval_multiply(fixed(float(axis[0])), vector[1]),
                    interval_multiply(fixed(float(axis[1])), vector[0]),
                ),
            )
            return (
                np.asarray(tuple(float(value[0]) for value in components)),
                np.asarray(tuple(float(value[1]) for value in components)),
            )

        radial_cross = cross_axis(radial_lower, radial_upper)
        tangent_cross = cross_axis(tangent_lower, tangent_upper)
        angle = (np.asarray(box[0, 0]), np.asarray(box[1, 0]))
        angle_cosine = array_interval(interval_cos(angle))
        angle_sine = array_interval(interval_sin(angle))
        negative_angle_sine = (-angle_sine[1], -angle_sine[0])
        first_column: list[tuple[np.ndarray, np.ndarray]] = []
        second_column: list[tuple[np.ndarray, np.ndarray]] = []
        for component in range(3):
            radial_component = (radial_lower[component], radial_upper[component])
            tangent_component = (tangent_lower[component], tangent_upper[component])
            first_column.append(
                array_interval(
                    interval_add(
                        interval_multiply(radial_component, negative_angle_sine),
                        interval_multiply(
                            (radial_cross[0][component], radial_cross[1][component]),
                            angle_cosine,
                        ),
                    )
                )
            )
            second_column.append(
                array_interval(
                    interval_add(
                        interval_add(
                            interval_multiply(
                                (
                                    np.asarray(axis[component]),
                                    np.asarray(axis[component]),
                                ),
                                axial_tangent,
                            ),
                            interval_multiply(tangent_component, angle_cosine),
                        ),
                        interval_multiply(
                            (tangent_cross[0][component], tangent_cross[1][component]),
                            angle_sine,
                        ),
                    )
                )
            )
        lower = np.column_stack(
            (
                tuple(float(value[0]) for value in first_column),
                tuple(float(value[0]) for value in second_column),
            )
        )
        upper = np.column_stack(
            (
                tuple(float(value[1]) for value in first_column),
                tuple(float(value[1]) for value in second_column),
            )
        )
        return lower, upper

    def exact_quadric_witness(
        self,
        profile_parameter_interval: tuple[float | Fraction, float | Fraction]
        | None = None,
        /,
    ) -> RevolutionQuadricWitness | None:
        """Return exact quadric data while retaining this authored source patch."""
        return _revolution_quadric_witness(self, profile_parameter_interval)

    def exact_sphere_revolution_witness(self, /) -> _ExactSphereRevolutionWitness | None:
        """Prove this circular meridian revolves to a source sphere."""
        return _exact_spherical_revolution_witness(self)


def _exact_dot3(
    first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
) -> ExactScalar:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)),
        Fraction(0),
    )


def _exact_cross3(
    first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _exact_unit3(
    vector: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar, ExactScalar] | None:
    squared_norm = _exact_dot3(vector, vector)
    if not isinstance(squared_norm, Fraction) or squared_norm <= 0:
        return None
    inverse_norm = RadicalElement.sqrt(squared_norm).reciprocal()
    return cast(
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple(component * inverse_norm for component in vector),
    )


@dataclass(frozen=True, slots=True, eq=False)
class RevolutionQuadricWitness:
    """Exact analytic view of a RevolutionSurface, bound to its source object."""

    source: RevolutionSurface
    family: Literal["plane", "cylinder", "cone", "sphere"]
    center_exact: tuple[ExactScalar, ...] | None
    origin_exact: tuple[ExactScalar, ...] | None
    axis_exact: tuple[ExactScalar, ...]
    radial_axis_exact: tuple[ExactScalar, ...]
    second_radial_axis_exact: tuple[ExactScalar, ...]
    radius_exact: ExactScalar | None
    radial_origin_exact: ExactScalar | None
    radial_direction_exact: ExactScalar | None
    axial_origin_exact: ExactScalar | None
    axial_direction_exact: ExactScalar | None
    dr_dz_exact: ExactScalar | None
    profile_parameter_interval: tuple[Fraction, Fraction]
    frame_sign: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, RevolutionSurface):
            raise TypeError("A quadric witness must retain its RevolutionSurface source.")
        vectors = (
            self.axis_exact,
            self.radial_axis_exact,
            self.second_radial_axis_exact,
        )
        points = (self.center_exact, self.origin_exact)
        scalars = (
            self.radius_exact,
            self.radial_origin_exact,
            self.radial_direction_exact,
            self.axial_origin_exact,
            self.axial_direction_exact,
            self.dr_dz_exact,
        )
        if (
            any(
                len(vector) != 3
                or any(
                    not isinstance(value, (Fraction, RadicalElement)) for value in vector
                )
                for vector in vectors
            )
            or any(
                point is not None
                and (
                    len(point) != 3
                    or any(
                        not isinstance(value, (Fraction, RadicalElement))
                        for value in point
                    )
                )
                for point in points
            )
            or any(
                value is not None and not isinstance(value, (Fraction, RadicalElement))
                for value in scalars
            )
        ):
            raise TypeError("Quadric witness coefficients must be exact source scalars.")
        if (
            _exact_dot3(self.axis_exact, self.axis_exact) != 1
            or _exact_dot3(self.radial_axis_exact, self.radial_axis_exact) != 1
            or _exact_dot3(self.second_radial_axis_exact, self.second_radial_axis_exact)
            != 1
            or _exact_dot3(self.axis_exact, self.radial_axis_exact) != 0
            or _exact_dot3(self.axis_exact, self.second_radial_axis_exact) != 0
            or _exact_dot3(self.radial_axis_exact, self.second_radial_axis_exact) != 0
            or _exact_cross3(self.axis_exact, self.radial_axis_exact)
            != self.second_radial_axis_exact
        ):
            raise ValueError("Quadric witness frame must be exactly orthonormal.")
        if (
            len(self.profile_parameter_interval) != 2
            or any(
                not isinstance(value, Fraction)
                for value in self.profile_parameter_interval
            )
            or self.profile_parameter_interval[0] >= self.profile_parameter_interval[1]
        ):
            raise ValueError(
                "Quadric witness profile interval must be exact and ordered."
            )
        if (
            _exact_profile_interval(self.source.curve, self.profile_parameter_interval)
            != self.profile_parameter_interval
        ):
            raise ValueError("Quadric witness profile interval is not source-valid.")
        if self.frame_sign not in (-1, 1):
            raise ValueError("Quadric witness frame_sign must be -1 or 1.")
        if self.family == "sphere":
            valid_family = (
                self.center_exact is not None
                and self.origin_exact is None
                and self.radius_exact is not None
                and self.radius_exact > 0
                and self.radial_origin_exact is None
                and self.radial_direction_exact is None
                and self.axial_origin_exact is None
                and self.axial_direction_exact is None
                and self.dr_dz_exact is None
            )
        elif self.family == "cylinder":
            valid_family = (
                self.center_exact is None
                and self.origin_exact is not None
                and self.radius_exact is not None
                and self.radius_exact > 0
                and self.radial_origin_exact == self.radius_exact
                and self.radial_direction_exact == 0
                and self.axial_origin_exact is not None
                and self.axial_direction_exact not in (None, 0)
                and self.dr_dz_exact is None
            )
        elif self.family == "cone":
            valid_family = (
                self.center_exact is None
                and self.origin_exact is not None
                and self.radius_exact is None
                and self.radial_origin_exact is not None
                and self.radial_direction_exact not in (None, 0)
                and self.axial_origin_exact is not None
                and self.axial_direction_exact not in (None, 0)
                and self.dr_dz_exact
                == self.radial_direction_exact / self.axial_direction_exact
            )
        else:
            valid_family = (
                self.family == "plane"
                and self.center_exact is None
                and self.origin_exact is not None
                and self.radius_exact is None
                and self.radial_origin_exact is not None
                and self.radial_direction_exact not in (None, 0)
                and self.axial_origin_exact is not None
                and self.axial_direction_exact == 0
                and self.dr_dz_exact is None
            )
        if not valid_family:
            raise ValueError("Quadric witness coefficients do not match its family.")


@dataclass(frozen=True, slots=True)
class _RevolutionLineProfile:
    origin: tuple[ExactScalar, ...]
    axis: tuple[ExactScalar, ...]
    axial_origin: ExactScalar
    axial_direction: ExactScalar
    radial_origin: tuple[ExactScalar, ...]
    radial_direction: tuple[ExactScalar, ...]


def _exact_profile_interval(
    profile: AbstractCurve,
    requested: tuple[float | Fraction, float | Fraction] | None,
    /,
) -> tuple[Fraction, Fraction] | None:
    interval = requested if requested is not None else profile.parameter_domain
    if interval is None:
        return None
    if len(interval) != 2:
        raise ValueError("profile_parameter_interval must contain two endpoints.")

    def as_fraction(value: float | Fraction, /) -> Fraction:
        if isinstance(value, bool):
            raise ValueError("Profile parameter endpoints must be finite scalars.")
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Profile parameter endpoints must be finite scalars.")
        return Fraction.from_float(numeric)

    first, last = map(as_fraction, interval)
    if first >= last:
        raise ValueError("profile_parameter_interval must be increasing.")
    profile.is_c1_on(float(first), float(last))
    return first, last


def _prepare_revolution_line_profile(
    surface: RevolutionSurface, profile: AbstractCurve, /
) -> _RevolutionLineProfile | None:
    from .._rational import rational_vector

    if isinstance(profile, LineCurve):
        profile_origin = profile.origin_exact
        profile_direction = profile.direction_exact
    elif isinstance(profile, ExactLineCurve):
        profile_origin = profile.first_exact
        profile_direction = tuple(
            last - first
            for first, last in zip(profile.first_exact, profile.last_exact, strict=True)
        )
    else:
        return None
    axis_values = rational_vector(np.asarray(surface.axis_direction))
    origin = rational_vector(np.asarray(surface.axis_origin))
    if len(axis_values) != 3 or len(origin) != 3:
        return None
    axis = (axis_values[0], axis_values[1], axis_values[2])
    if len(profile_origin) != 3 or len(profile_direction) != 3:
        return None
    if _exact_dot3(axis, axis) != 1:
        return None
    relative = tuple(profile_origin[index] - origin[index] for index in range(3))
    axial_origin = _exact_dot3(axis, relative)
    axial_direction = _exact_dot3(axis, profile_direction)
    radial_origin = tuple(
        relative[index] - axis[index] * axial_origin for index in range(3)
    )
    radial_direction = tuple(
        profile_direction[index] - axis[index] * axial_direction for index in range(3)
    )
    if any(_exact_cross3(radial_origin, radial_direction)):
        return None
    return _RevolutionLineProfile(
        origin=origin,
        axis=axis,
        axial_origin=axial_origin,
        axial_direction=axial_direction,
        radial_origin=radial_origin,
        radial_direction=radial_direction,
    )


def _sphere_revolution_quadric_witness(
    surface: RevolutionSurface,
    profile: CircleCurve,
    profile_interval: tuple[Fraction, Fraction],
    /,
) -> RevolutionQuadricWitness | None:
    sphere = _exact_spherical_revolution_witness(surface)
    if sphere is None:
        return None
    center, radius, axis, frame_sign = sphere
    radial_candidates = tuple(
        tuple(
            candidate[index] - axis[index] * _exact_dot3(axis, candidate)
            for index in range(3)
        )
        for candidate in (profile.first_axis_exact, profile.second_axis_exact)
    )
    radial = _exact_unit3(radial_candidates[0])
    if radial is None:
        radial = _exact_unit3(radial_candidates[1])
    if radial is None:
        return None
    return RevolutionQuadricWitness(
        source=surface,
        family="sphere",
        center_exact=center,
        origin_exact=None,
        axis_exact=axis,
        radial_axis_exact=radial,
        second_radial_axis_exact=_exact_cross3(axis, radial),
        radius_exact=radius,
        radial_origin_exact=None,
        radial_direction_exact=None,
        axial_origin_exact=None,
        axial_direction_exact=None,
        dr_dz_exact=None,
        profile_parameter_interval=profile_interval,
        frame_sign=frame_sign,
    )


def _line_revolution_quadric_witness(
    surface: RevolutionSurface,
    line: _RevolutionLineProfile,
    profile_interval: tuple[Fraction, Fraction],
    /,
) -> RevolutionQuadricWitness | None:
    origin_nonzero = any(value != 0 for value in line.radial_origin)
    direction_nonzero = any(value != 0 for value in line.radial_direction)
    if not origin_nonzero and not direction_nonzero:
        return None
    radial = _exact_unit3(line.radial_origin if origin_nonzero else line.radial_direction)
    if radial is None:
        return None
    transverse = _exact_cross3(line.axis, radial)
    radial_origin = _exact_dot3(line.radial_origin, radial)
    radial_direction = _exact_dot3(line.radial_direction, radial)
    if line.axial_direction == 0:
        if not direction_nonzero:
            return None
        return RevolutionQuadricWitness(
            source=surface,
            family="plane",
            center_exact=None,
            origin_exact=line.origin,
            axis_exact=line.axis,
            radial_axis_exact=radial,
            second_radial_axis_exact=transverse,
            radius_exact=None,
            radial_origin_exact=radial_origin,
            radial_direction_exact=radial_direction,
            axial_origin_exact=line.axial_origin,
            axial_direction_exact=Fraction(0),
            dr_dz_exact=None,
            profile_parameter_interval=profile_interval,
            frame_sign=1,
        )
    if not direction_nonzero:
        if not origin_nonzero:
            return None
        return RevolutionQuadricWitness(
            source=surface,
            family="cylinder",
            center_exact=None,
            origin_exact=line.origin,
            axis_exact=line.axis,
            radial_axis_exact=radial,
            second_radial_axis_exact=transverse,
            radius_exact=radial_origin,
            radial_origin_exact=radial_origin,
            radial_direction_exact=Fraction(0),
            axial_origin_exact=line.axial_origin,
            axial_direction_exact=line.axial_direction,
            dr_dz_exact=None,
            profile_parameter_interval=profile_interval,
            frame_sign=1,
        )
    return RevolutionQuadricWitness(
        source=surface,
        family="cone",
        center_exact=None,
        origin_exact=line.origin,
        axis_exact=line.axis,
        radial_axis_exact=radial,
        second_radial_axis_exact=transverse,
        radius_exact=None,
        radial_origin_exact=radial_origin,
        radial_direction_exact=radial_direction,
        axial_origin_exact=line.axial_origin,
        axial_direction_exact=line.axial_direction,
        dr_dz_exact=radial_direction / line.axial_direction,
        profile_parameter_interval=profile_interval,
        frame_sign=1,
    )


def _revolution_quadric_witness(
    surface: RevolutionSurface,
    profile_parameter_interval: tuple[float | Fraction, float | Fraction] | None,
    /,
) -> RevolutionQuadricWitness | None:
    profile = surface.curve
    exact_interval = _exact_profile_interval(profile, profile_parameter_interval)
    if exact_interval is None:
        return None
    if isinstance(profile, CircleCurve):
        return _sphere_revolution_quadric_witness(surface, profile, exact_interval)
    line = _prepare_revolution_line_profile(surface, profile)
    if line is None:
        return None
    return _line_revolution_quadric_witness(surface, line, exact_interval)


def _exact_spherical_revolution_witness(
    surface: RevolutionSurface, /
) -> _ExactSphereRevolutionWitness | None:
    profile = surface.curve
    if not isinstance(profile, CircleCurve) or profile.ambient_dimension != 3:
        return None
    center = profile.center_exact
    first_axis = profile.first_axis_exact
    second_axis = profile.second_axis_exact
    if len(center) != 3 or len(first_axis) != 3 or len(second_axis) != 3:
        return None

    from .._rational import rational_vector

    origin = rational_vector(np.asarray(surface.axis_origin))
    axis_values = rational_vector(np.asarray(surface.axis_direction))
    if len(origin) != 3 or len(axis_values) != 3:
        return None
    axis = (axis_values[0], axis_values[1], axis_values[2])

    def dot(
        first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    normal = (
        first_axis[1] * second_axis[2] - first_axis[2] * second_axis[1],
        first_axis[2] * second_axis[0] - first_axis[0] * second_axis[2],
        first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0],
    )
    displacement = (
        center[0] - origin[0],
        center[1] - origin[1],
        center[2] - origin[2],
    )
    cross_axis_displacement = (
        displacement[1] * axis[2] - displacement[2] * axis[1],
        displacement[2] * axis[0] - displacement[0] * axis[2],
        displacement[0] * axis[1] - displacement[1] * axis[0],
    )
    if (
        dot(first_axis, first_axis) != 1
        or dot(second_axis, second_axis) != 1
        or dot(first_axis, second_axis) != 0
        or dot(axis, axis) != 1
        or dot(axis, normal) != 0
        or any(cross_axis_displacement)
        or profile.radius_exact <= 0
    ):
        return None
    return center, profile.radius_exact, axis, 1


class SurfaceMappedCurve(AbstractCurve):
    """Exact parameterized curve obtained by evaluating a 2D source curve on a surface."""

    surface: AbstractSurfacePatch
    pcurve: AbstractCurve

    def __init__(self, surface: AbstractSurfacePatch, pcurve: AbstractCurve, /) -> None:
        if not isinstance(surface, AbstractSurfacePatch):
            raise TypeError("SurfaceMappedCurve.surface must be an AbstractSurfacePatch.")
        if not isinstance(pcurve, AbstractCurve) or pcurve.ambient_dimension != 2:
            raise TypeError("SurfaceMappedCurve.pcurve must be a 2D AbstractCurve.")
        self.surface = surface
        self.pcurve = pcurve

    def identity_payload(self, /) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "surface-mapped-curve",
            "surface": carrier_payload(self.surface),
            "pcurve": carrier_payload(self.pcurve),
        }

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        period = self.pcurve.period
        if period is not None:
            return period
        return _surface_mapped_native_period(self.surface, self.pcurve)[0]

    def native_period_symbol(self, /) -> str | None:
        if self.pcurve.period is not None:
            return self.pcurve.native_period_symbol()
        return _surface_mapped_native_period(self.surface, self.pcurve)[1]

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.pcurve.parameter_domain

    def evaluate(self, parameters: Array, /) -> Array:
        return self.surface.evaluate(self.pcurve.evaluate(parameters))

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._interval_enclosure import interval_cos, interval_sin
        from .._rational import fraction_down, fraction_up, rational_vector

        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("SurfaceMappedCurve bounds must be finite and ordered.")
        domain = self.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            raise ValueError(
                "SurfaceMappedCurve bounds must lie in its source parameter domain."
            )
        pcurve_box = np.asarray(self.pcurve.bounding_box(first, last), dtype=np.float64)
        if (
            pcurve_box.shape != (2, 2)
            or not np.all(np.isfinite(pcurve_box))
            or np.any(pcurve_box[0] > pcurve_box[1])
        ):
            raise ValueError(
                "SurfaceMappedCurve p-curve has no finite ordered interval box."
            )
        chart = tuple(
            (
                Fraction.from_float(float(pcurve_box[0, axis])),
                Fraction.from_float(float(pcurve_box[1, axis])),
            )
            for axis in range(2)
        )

        def interval_add(
            left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], /
        ) -> tuple[Fraction, Fraction]:
            return left[0] + right[0], left[1] + right[1]

        def interval_multiply(
            left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], /
        ) -> tuple[Fraction, Fraction]:
            products = tuple(a * b for a in left for b in right)
            return min(products), max(products)

        def coefficient(value: ExactScalar, /) -> tuple[Fraction, Fraction]:
            return coefficient_interval(value, bits=128)

        def trig(
            function: Callable[[Interval], Interval], value: tuple[Fraction, Fraction], /
        ) -> tuple[Fraction, Fraction]:
            lower, upper = function(
                (
                    np.asarray(float(value[0]), dtype=np.float64),
                    np.asarray(float(value[1]), dtype=np.float64),
                )
            )
            return Fraction.from_float(float(lower)), Fraction.from_float(float(upper))

        def realize_box(
            lower: tuple[Fraction, Fraction, Fraction],
            upper: tuple[Fraction, Fraction, Fraction],
            /,
        ) -> np.ndarray:
            return np.asarray(
                (
                    tuple(fraction_down(value) for value in lower),
                    tuple(fraction_up(value) for value in upper),
                ),
                dtype=np.float64,
            )

        def plane_box(
            origin_values: tuple[ExactScalar, ...],
            first_values: tuple[ExactScalar, ...],
            second_values: tuple[ExactScalar, ...],
            /,
        ) -> np.ndarray:
            lower: list[Fraction] = []
            upper: list[Fraction] = []
            for axis in range(3):
                value = coefficient(origin_values[axis])
                value = interval_add(
                    value,
                    interval_multiply(coefficient(first_values[axis]), chart[0]),
                )
                value = interval_add(
                    value,
                    interval_multiply(coefficient(second_values[axis]), chart[1]),
                )
                lower.append(value[0])
                upper.append(value[1])
            return realize_box(
                (lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2])
            )

        def cylinder_box(patch: CylinderPatch, radius: ExactScalar, /) -> np.ndarray:
            if (
                isinstance(self.pcurve, ConicRootLinePCurve)
                and self.pcurve.phase_shift_exact == 0
                and isinstance(self.pcurve.root.conic, CircleCurve)
            ):
                conic = self.pcurve.root.conic
                if (
                    conic.center_exact == patch.origin_exact
                    and conic.first_axis_exact == patch.first_axis_exact
                    and conic.second_axis_exact == patch.second_axis_exact
                    and conic.radius_exact == radius
                ):
                    parameters = tuple(
                        Fraction.from_float(float(value)) for value in (first, last)
                    )
                    heights = tuple(
                        self.pcurve.height_origin_exact
                        + self.pcurve.height_direction_exact * value
                        for value in parameters
                    )
                    height = (
                        coefficient(min(heights))[0],
                        coefficient(max(heights))[1],
                    )
                    root_point = self.pcurve.root.exact_point()
                    lower: list[Fraction] = []
                    upper: list[Fraction] = []
                    for point_value, axis_value in zip(
                        root_point, patch.axis_exact, strict=True
                    ):
                        interval = interval_add(
                            coefficient(point_value),
                            interval_multiply(coefficient(axis_value), height),
                        )
                        lower.append(interval[0])
                        upper.append(interval[1])
                    return realize_box(
                        (lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2])
                    )
            cosine = trig(interval_cos, chart[0])
            sine = trig(interval_sin, chart[0])
            lower, upper = [], []
            for axis in range(3):
                radial = interval_add(
                    interval_multiply(
                        coefficient(radius * patch.first_axis_exact[axis]), cosine
                    ),
                    interval_multiply(
                        coefficient(radius * patch.second_axis_exact[axis]), sine
                    ),
                )
                height = interval_multiply(coefficient(patch.axis_exact[axis]), chart[1])
                point = interval_add(coefficient(patch.origin_exact[axis]), radial)
                point = interval_add(point, height)
                lower.append(point[0])
                upper.append(point[1])
            return realize_box(
                (lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2])
            )

        surface = self.surface
        if isinstance(surface, PlanePatch):
            return plane_box(
                rational_vector(np.asarray(surface.origin)),
                rational_vector(np.asarray(surface.first_axis)),
                rational_vector(np.asarray(surface.second_axis)),
            )
        if isinstance(surface, CylinderPatch):
            return cylinder_box(surface, surface.radius_exact)
        if isinstance(surface, OffsetSurface) and isinstance(surface.base, CylinderPatch):
            base = surface.base
            return cylinder_box(
                base,
                base.radius_exact + surface.frame_sign * surface.distance_exact,
            )
        if isinstance(surface, ExtrusionSurface):
            profile_box = np.asarray(
                surface.curve.bounding_box(float(chart[0][0]), float(chart[0][1])),
                dtype=np.float64,
            )
            if (
                profile_box.shape != (2, 3)
                or not np.all(np.isfinite(profile_box))
                or np.any(profile_box[0] > profile_box[1])
            ):
                raise ValueError(
                    "The extrusion profile has no finite ordered source enclosure."
                )
            profile = tuple(
                (
                    Fraction.from_float(float(profile_box[0, axis])),
                    Fraction.from_float(float(profile_box[1, axis])),
                )
                for axis in range(3)
            )
            direction = rational_vector(np.asarray(surface.direction))
            lower: list[Fraction] = []
            upper: list[Fraction] = []
            for axis in range(3):
                value = interval_add(
                    profile[axis],
                    interval_multiply((direction[axis], direction[axis]), chart[1]),
                )
                lower.append(value[0])
                upper.append(value[1])
            return realize_box(
                (lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2])
            )
        if isinstance(surface, (RevolutionSurface, OffsetSurface)):
            from ._model import _bspline_revolution_box

            box = _bspline_revolution_box(
                surface,
                (float(pcurve_box[0, 0]), float(pcurve_box[0, 1])),
                (float(pcurve_box[1, 0]), float(pcurve_box[1, 1])),
            )
            if box is not None:
                return np.stack(box)
        # Isoline p-curves have a zero-width chart axis. A one-ulp widening inside
        # the patch domain is a superset box, so the enclosure stays sound.
        surface_box = np.array(pcurve_box, dtype=np.float64)
        surface_domain = surface.parameter_domain
        for axis in range(2):
            if surface_box[0, axis] < surface_box[1, axis]:
                continue
            upper_limit = (
                np.inf if surface_domain is None else float(surface_domain[1][axis])
            )
            if surface_box[1, axis] < upper_limit:
                surface_box[1, axis] = np.nextafter(surface_box[1, axis], np.inf)
            else:
                surface_box[0, axis] = np.nextafter(surface_box[0, axis], -np.inf)
        source_box = np.asarray(surface.bounding_box(surface_box), dtype=np.float64)
        if (
            source_box.shape != (2, 3)
            or not np.all(np.isfinite(source_box))
            or np.any(source_box[0] > source_box[1])
        ):
            raise ValueError(
                f"{type(surface).__name__}.bounding_box must return a finite ordered 3D box."
            )
        return source_box

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("SurfaceMappedCurve derivative bounds support order=1 only.")
        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError(
                "SurfaceMappedCurve derivative bounds must be finite and ordered."
            )
        if not self.pcurve.is_c1_on(first, last):
            raise NotImplementedError(
                "SurfaceMappedCurve p-curve is not C1 on the requested interval."
            )
        velocity = _linear_pcurve_velocity(self.pcurve)
        if not isinstance(self.surface, RevolutionSurface) or velocity is None:
            raise NotImplementedError(
                "Source-backed derivative bounds are unavailable for this mapped curve action."
            )
        chart_box = np.asarray(self.pcurve.bounding_box(first, last), dtype=np.float64)
        surface_lower, surface_upper = self.surface.derivative_bounds(chart_box)
        from .._interval_enclosure import interval_add, interval_multiply
        from .._rational import fraction_down, fraction_up

        lower = np.zeros((3,), dtype=np.float64)
        upper = np.zeros((3,), dtype=np.float64)
        for axis, coefficient in enumerate(velocity):
            coefficient_lower, coefficient_upper = coefficient_interval(
                coefficient, bits=128
            )
            scalar = (
                np.asarray(fraction_down(coefficient_lower), dtype=np.float64),
                np.asarray(fraction_up(coefficient_upper), dtype=np.float64),
            )
            term = interval_multiply(
                (surface_lower[:, axis], surface_upper[:, axis]),
                (np.broadcast_to(scalar[0], (3,)), np.broadcast_to(scalar[1], (3,))),
            )
            accumulated = interval_add((lower, upper), term)
            lower, upper = np.asarray(accumulated[0]), np.asarray(accumulated[1])
        return lower, upper

    def is_c1_on(self, first: float, last: float, /) -> bool:
        if not self.pcurve.is_c1_on(first, last):
            return False
        if isinstance(self.surface, RevolutionSurface) and (
            _linear_pcurve_velocity(self.pcurve) is not None
        ):
            chart_box = np.asarray(
                self.pcurve.bounding_box(first, last), dtype=np.float64
            )
            return self.surface.is_c1_on(chart_box)
        if isinstance(self.pcurve, ExactLineCurve):
            origin = np.asarray(self.pcurve.first_exact, dtype=np.float64)
            direction = np.asarray(
                tuple(
                    right - left
                    for left, right in zip(
                        self.pcurve.first_exact, self.pcurve.last_exact, strict=True
                    )
                ),
                dtype=np.float64,
            )
            fixed = tuple(
                axis
                for axis, (left, right) in enumerate(
                    zip(self.pcurve.first_exact, self.pcurve.last_exact, strict=True)
                )
                if left == right
            )
        elif isinstance(self.pcurve, LineCurve):
            origin = np.asarray(self.pcurve.origin, dtype=np.float64)
            direction = np.asarray(self.pcurve.direction, dtype=np.float64)
            fixed = tuple(np.flatnonzero(direction == 0.0).tolist())
        else:
            return False
        if len(fixed) != 1:
            return False
        fixed_axis = fixed[0]
        free_axis = 1 - fixed_axis
        if direction[free_axis] == 0.0:
            return False
        first_free = origin[free_axis] + direction[free_axis] * first
        last_free = origin[free_axis] + direction[free_axis] * last
        return self.surface.is_c1_along(
            fixed_axis,
            float(origin[fixed_axis]),
            min(first_free, last_free),
            max(first_free, last_free),
        )


class SurfaceIsoparametricCurve(AbstractCurve):
    """Source seam or isoline with one surface coordinate held fixed."""

    surface: AbstractSurfacePatch
    fixed_axis: int = eqx.field(static=True)
    fixed_value: float = eqx.field(static=True)
    fixed_value_exact: ExactScalar | None = eqx.field(static=True)
    parameter_range: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        surface: AbstractSurfacePatch,
        fixed_axis: int,
        fixed_value: float,
        /,
        *,
        fixed_value_exact: ExactScalar | None = None,
        parameter_range: tuple[float, float] | None = None,
    ) -> None:
        if not isinstance(surface, AbstractSurfacePatch):
            raise TypeError("SurfaceIsoparametricCurve.surface must be a surface patch.")
        if isinstance(fixed_axis, bool) or not isinstance(fixed_axis, int):
            raise TypeError("SurfaceIsoparametricCurve.fixed_axis must be an integer.")
        if fixed_axis not in (0, 1):
            raise ValueError("SurfaceIsoparametricCurve.fixed_axis must be 0 or 1.")
        if not np.isfinite(fixed_value):
            raise ValueError("SurfaceIsoparametricCurve.fixed_value must be finite.")
        if fixed_value_exact is not None:
            if not isinstance(fixed_value_exact, (Fraction, RadicalElement)):
                raise TypeError(
                    "SurfaceIsoparametricCurve.fixed_value_exact must be an exact scalar."
                )
            if float(fixed_value_exact) != float(fixed_value):
                raise ValueError(
                    "SurfaceIsoparametricCurve exact fixed value must match its execution shadow."
                )
        if parameter_range is not None:
            first, last = parameter_range
            if not np.isfinite(first) or not np.isfinite(last) or first >= last:
                raise ValueError("parameter_range must be finite and increasing.")
            parameter_range = (float(first), float(last))
        self.surface = surface
        self.fixed_axis = fixed_axis
        self.fixed_value = float(fixed_value)
        self.fixed_value_exact = fixed_value_exact
        self.parameter_range = parameter_range

    def identity_payload(self) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "surface-isoparametric-curve",
            "surface": carrier_payload(self.surface),
            "fixed_axis": self.fixed_axis,
            "fixed_value": float(self.fixed_value).hex(),
            "fixed_value_exact": (
                None
                if self.fixed_value_exact is None
                else coefficient_payload(self.fixed_value_exact)
            ),
            "parameter_range": (
                None
                if self.parameter_range is None
                else tuple(float(value).hex() for value in self.parameter_range)
            ),
        }

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def period(self) -> float | None:
        return self.surface.periods[1 - self.fixed_axis]

    def native_period_symbol(self, /) -> str | None:
        return self.surface.native_period_symbols()[1 - self.fixed_axis]

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        if self.parameter_range is not None:
            return self.parameter_range
        domain = self.surface.parameter_domain
        if domain is not None:
            return domain[0][1 - self.fixed_axis], domain[1][1 - self.fixed_axis]
        if self.period is not None:
            return 0.0, self.period
        return None

    def evaluate(self, parameters: Array, /) -> Array:
        parameter = jnp.asarray(parameters, dtype=jnp.float64)
        fixed = jnp.full_like(parameter, self.fixed_value)
        coordinates = (
            jnp.stack((fixed, parameter), axis=-1)
            if self.fixed_axis == 0
            else jnp.stack((parameter, fixed), axis=-1)
        )
        return self.surface.evaluate(coordinates)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("Curve ranges must be finite and nondecreasing.")
        domain = self.parameter_domain
        period = self.period
        if not (
            domain is not None
            and self.native_period_symbol() == "two_pi"
            and period is not None
            and domain[1] == period
            and domain[0] <= first <= domain[1] < last
        ):
            return super().bounding_box(first, last)

        from .._interval_enclosure import prepare_interval_function
        from .._rational import fraction_up
        from ._trim_helpers import period_scalar_bounds, PeriodOffset

        period_lower, period_upper = period_scalar_bounds(
            PeriodOffset(Fraction(0), Fraction(1))
        )
        wrapped_upper = float(
            fraction_up(
                Fraction.from_float(last) - Fraction.from_float(float(period_lower))
            )
        )
        if wrapped_upper > domain[1]:
            raise ValueError("Periodic source enclosure crosses more than one turn.")

        def value(parameter: Array, /) -> Array:
            return self.evaluate(parameter[0])

        prepared = prepare_interval_function(value, 1)
        left_lower, left_upper = prepared.evaluate(
            np.asarray(((first,),), dtype=np.float64),
            np.asarray(((float(period_upper),),), dtype=np.float64),
        )
        right_lower, right_upper = prepared.evaluate(
            np.asarray(((0.0,),), dtype=np.float64),
            np.asarray(((wrapped_upper,),), dtype=np.float64),
        )
        return np.stack(
            (
                np.minimum(left_lower[0], right_lower[0]),
                np.maximum(left_upper[0], right_upper[0]),
            )
        )

    def is_c1_on(self, first: float, last: float, /) -> bool:
        self.validate_query_range(first, last)
        return self.surface.is_c1_along(
            self.fixed_axis,
            self.fixed_value,
            first,
            last,
        )

    def validate_query_range(self, first: float, last: float, /) -> None:
        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("Curve ranges must be finite and nondecreasing.")
        domain = self.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            raise ValueError("Curve range must lie in its source parameter domain.")


def _conic_parameter_root_bounds(
    cosine: ExactScalar, sine: ExactScalar, turns: int, /
) -> tuple[float, float]:
    """Isolate an exact algebraic unit direction in its native angular chart."""
    from .._interval_enclosure import interval_cos, interval_sin
    from .._rational import fraction_down, fraction_up
    from ._trim_helpers import period_scalar_bounds, PeriodOffset

    def point_bounds(
        function: Callable[[Interval], Interval], parameter: float
    ) -> tuple[Fraction, Fraction]:
        lower, upper = function(
            (
                np.asarray(parameter, dtype=np.float64),
                np.asarray(parameter, dtype=np.float64),
            )
        )
        return Fraction.from_float(float(lower)), Fraction.from_float(float(upper))

    def product(
        first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
    ) -> tuple[Fraction, Fraction]:
        values = tuple(a * b for a in first for b in second)
        return min(values), max(values)

    def angular_side(parameter: float, bits: int, /) -> int:
        cos_lower, cos_upper = point_bounds(interval_cos, parameter)
        sin_lower, sin_upper = point_bounds(interval_sin, parameter)
        root_cosine = coefficient_interval(cosine, bits=bits)
        root_sine = coefficient_interval(sine, bits=bits)
        cross_cosine = product(root_cosine, (sin_lower, sin_upper))
        cross_sine = product(root_sine, (cos_lower, cos_upper))
        dot_cosine = product(root_cosine, (cos_lower, cos_upper))
        dot_sine = product(root_sine, (sin_lower, sin_upper))
        cross_lower = cross_cosine[0] - cross_sine[1]
        cross_upper = cross_cosine[1] - cross_sine[0]
        dot_lower = dot_cosine[0] + dot_sine[0]
        if dot_lower <= 0:
            return 0
        if cross_upper < 0:
            return -1
        if cross_lower > 0:
            return 1
        return 0

    def left_of_root(parameter: float, bits: int, /) -> bool:
        return angular_side(parameter, bits) < 0

    def right_of_root(parameter: float, bits: int, /) -> bool:
        return angular_side(parameter, bits) > 0

    if cosine == 1 and sine == 0:
        principal_lower = principal_upper = Fraction(0)
    else:
        center = atan2(float(sine), float(cosine))
        if center < 0:
            center += tau
        radius = max(ulp(center), float(np.nextafter(0.0, np.inf)))
        bits = 80
        while True:
            lower = float(np.nextafter(center - radius, -np.inf))
            upper = float(np.nextafter(center + radius, np.inf))
            if left_of_root(lower, bits) and right_of_root(upper, bits):
                principal_lower = Fraction.from_float(lower)
                principal_upper = Fraction.from_float(upper)
                break
            next_radius = radius * 2.0
            if (
                not np.isfinite(next_radius)
                or next_radius >= tau
                or next_radius <= radius
            ):
                raise ValueError(
                    "The conic parameter root could not be isolated at binary64 resolution."
                )
            radius = next_radius
            bits *= 2

    if turns == 0:
        turns_bound = (0.0, 0.0)
    else:
        turns_bound = period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(turns)))
    lower_exact = principal_lower + Fraction.from_float(float(turns_bound[0]))
    upper_exact = principal_upper + Fraction.from_float(float(turns_bound[1]))
    return fraction_down(lower_exact), fraction_up(upper_exact)


class ConicParameterRoot(StrictModule):
    """Exact algebraic unit direction selecting one point on a conic carrier."""

    conic: CircleCurve | EllipseCurve
    cosine_exact: ExactScalar = eqx.field(static=True)
    sine_exact: ExactScalar = eqx.field(static=True)
    turns: int = eqx.field(static=True)
    parameter_lower: float = eqx.field(static=True)
    parameter_upper: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    def __init__(
        self,
        conic: CircleCurve | EllipseCurve,
        cosine_exact: ExactScalar,
        sine_exact: ExactScalar,
        /,
        *,
        turns: int = 0,
    ) -> None:
        if not isinstance(conic, (CircleCurve, EllipseCurve)):
            raise TypeError(
                "ConicParameterRoot.conic must be a CircleCurve or EllipseCurve."
            )
        if not isinstance(cosine_exact, (Fraction, RadicalElement)) or not isinstance(
            sine_exact, (Fraction, RadicalElement)
        ):
            raise TypeError(
                "ConicParameterRoot radial coefficients must be ExactScalar values."
            )
        try:
            squared_norm = rational_coefficient(
                cosine_exact * cosine_exact + sine_exact * sine_exact,
                role="conic root unit radial norm",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "ConicParameterRoot requires an exact algebraic unit radial direction."
            ) from error
        if squared_norm != 1:
            raise ValueError(
                "ConicParameterRoot requires an exact algebraic unit radial direction."
            )
        if isinstance(turns, bool) or not isinstance(turns, int):
            raise TypeError("ConicParameterRoot.turns must be an integer.")
        lower, upper = _conic_parameter_root_bounds(cosine_exact, sine_exact, turns)
        self.conic = conic
        self.cosine_exact, self.sine_exact = cosine_exact, sine_exact
        self.turns = turns
        self.parameter_lower, self.parameter_upper = lower, upper
        self.root_id = canonical_fingerprint(self.identity_payload())

    @classmethod
    def from_exact_point(
        cls,
        conic: CircleCurve | EllipseCurve,
        point_exact: Sequence[ExactScalar],
        angle_hint: float,
        /,
    ) -> ConicParameterRoot | None:
        """Recover a conic root from its exact source point and chart hint.

        The hint selects the periodic turn only; the root definition is derived
        exclusively from exact conic and point coefficients.
        """
        try:
            hint = float(angle_hint)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not isinstance(conic, (CircleCurve, EllipseCurve))
            or conic.ambient_dimension != 3
            or not np.isfinite(hint)
            or not isinstance(point_exact, Sequence)
            or len(point_exact) != 3
            or any(
                not isinstance(value, (Fraction, RadicalElement)) for value in point_exact
            )
        ):
            return None

        def exact_dot(
            first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
        ) -> ExactScalar:
            return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

        first_axis, second_axis = conic.first_axis_exact, conic.second_axis_exact
        gram = (
            (exact_dot(first_axis, first_axis), exact_dot(first_axis, second_axis)),
            (exact_dot(second_axis, first_axis), exact_dot(second_axis, second_axis)),
        )
        if gram != ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))):
            return None
        delta = tuple(
            value - center
            for value, center in zip(point_exact, conic.center_exact, strict=True)
        )
        if isinstance(conic, CircleCurve):
            first_radius = second_radius = conic.radius_exact
        else:
            first_radius, second_radius = (
                conic.first_radius_exact,
                conic.second_radius_exact,
            )
        cosine = exact_dot(delta, first_axis) / first_radius
        sine = exact_dot(delta, second_axis) / second_radius
        principal = atan2(float(sine), float(cosine))
        if principal < 0.0:
            principal += tau
        turns = int(round((hint - principal) / tau))
        try:
            root = cls(conic, cosine, sine, turns=turns)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        if not root.parameter_lower <= hint <= root.parameter_upper:
            return None
        return root

    @property
    def parameter(self) -> float:
        principal = atan2(float(self.sine_exact), float(self.cosine_exact))
        if principal < 0.0:
            principal += tau
        value = principal + self.turns * tau
        return min(max(value, self.parameter_lower), self.parameter_upper)

    def parameter_enclosure(self, /) -> tuple[float, float]:
        return self.parameter_lower, self.parameter_upper

    def exact_point(self, /) -> tuple[ExactScalar, ...]:
        if isinstance(self.conic, CircleCurve):
            first_radius = second_radius = self.conic.radius_exact
        else:
            first_radius, second_radius = (
                self.conic.first_radius_exact,
                self.conic.second_radius_exact,
            )
        return tuple(
            self.conic.center_exact[axis]
            + first_radius * self.cosine_exact * self.conic.first_axis_exact[axis]
            + second_radius * self.sine_exact * self.conic.second_axis_exact[axis]
            for axis in range(self.conic.ambient_dimension)
        )

    def identity_payload(self) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload

        return {
            "kind": "exact-conic-parameter-root",
            "conic": carrier_payload(self.conic),
            "cosine": coefficient_payload(self.cosine_exact),
            "sine": coefficient_payload(self.sine_exact),
            "turns": self.turns,
        }


def _combine_conic_parameter_roots(
    first: ConicParameterRoot,
    second: ConicParameterRoot,
    /,
    *,
    subtract: bool = False,
) -> ConicParameterRoot | None:
    from ._trim_helpers import period_scalar_bounds, PeriodOffset

    cosine = (
        first.cosine_exact * second.cosine_exact + first.sine_exact * second.sine_exact
        if subtract
        else first.cosine_exact * second.cosine_exact
        - first.sine_exact * second.sine_exact
    )
    sine = (
        first.sine_exact * second.cosine_exact - first.cosine_exact * second.sine_exact
        if subtract
        else first.sine_exact * second.cosine_exact
        + first.cosine_exact * second.sine_exact
    )
    try:
        first_principal = ConicParameterRoot(
            first.conic, first.cosine_exact, first.sine_exact
        )
        second_principal = ConicParameterRoot(
            second.conic, second.cosine_exact, second.sine_exact
        )
        combined_principal = ConicParameterRoot(first.conic, cosine, sine)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None

    first_lower = Fraction.from_float(first_principal.parameter_lower)
    first_upper = Fraction.from_float(first_principal.parameter_upper)
    second_lower = Fraction.from_float(second_principal.parameter_lower)
    second_upper = Fraction.from_float(second_principal.parameter_upper)
    if subtract:
        total_lower, total_upper = (
            first_lower - second_upper,
            first_upper - second_lower,
        )
        candidates = (-1, 0)
    else:
        total_lower, total_upper = (
            first_lower + second_lower,
            first_upper + second_upper,
        )
        candidates = (0, 1)

    principal_lower = Fraction.from_float(combined_principal.parameter_lower)
    principal_upper = Fraction.from_float(combined_principal.parameter_upper)
    matched_turns: list[int] = []
    for candidate in candidates:
        period_lower, period_upper = period_scalar_bounds(
            PeriodOffset(Fraction(0), Fraction(candidate))
        )
        candidate_lower = principal_lower + Fraction.from_float(float(period_lower))
        candidate_upper = principal_upper + Fraction.from_float(float(period_upper))
        if total_lower <= candidate_upper and candidate_lower <= total_upper:
            matched_turns.append(candidate)
    if len(matched_turns) != 1:
        return None
    turns = first.turns + (-second.turns if subtract else second.turns)
    try:
        return ConicParameterRoot(
            first.conic,
            cosine,
            sine,
            turns=turns + matched_turns[0],
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _conic_root_phase_enclosure(
    root: ConicParameterRoot, phase_shift: ExactScalar, /
) -> tuple[float, float]:
    from .._rational import fraction_down, fraction_up

    shift_lower, shift_upper = coefficient_interval(phase_shift, bits=128)
    return (
        fraction_down(Fraction.from_float(root.parameter_lower) + shift_lower),
        fraction_up(Fraction.from_float(root.parameter_upper) + shift_upper),
    )


def _linear_pcurve_velocity(
    curve: AbstractCurve, /
) -> tuple[ExactScalar, ExactScalar] | None:
    from ._boolean_core import AffinePCurve, PeriodicPCurve

    if isinstance(curve, ConicRootLinePCurve):
        return Fraction(0), curve.height_direction_exact
    if isinstance(curve, ExactLineCurve):
        if len(curve.first_exact) != 2 or len(curve.last_exact) != 2:
            return None
        return (
            curve.last_exact[0] - curve.first_exact[0],
            curve.last_exact[1] - curve.first_exact[1],
        )
    if isinstance(curve, LineCurve):
        if len(curve.direction_exact) != 2:
            return None
        return curve.direction_exact[0], curve.direction_exact[1]
    if isinstance(curve, AffinePCurve):
        velocity = _linear_pcurve_velocity(curve.curve)
        if velocity is None:
            return None
        return (
            curve.matrix[0][0] * velocity[0] + curve.matrix[0][1] * velocity[1],
            curve.matrix[1][0] * velocity[0] + curve.matrix[1][1] * velocity[1],
        )
    if isinstance(curve, PeriodicPCurve):
        return _linear_pcurve_velocity(curve.source_curve)
    return None


def _surface_mapped_native_period(
    surface: AbstractSurfacePatch, pcurve: AbstractCurve, /
) -> tuple[float | None, str | None]:
    periods = surface.periods
    symbols = surface.native_period_symbols()
    if not any(period is not None for period in periods):
        return None, None
    velocity = _linear_pcurve_velocity(pcurve)
    if velocity is None:
        raise NotImplementedError(
            f"Native period derivation is unavailable for {type(pcurve).__name__} actions."
        )
    active = tuple(axis for axis, value in enumerate(velocity) if value != 0)
    if not active:
        return None, None
    rates: list[Fraction] = []
    for axis in active:
        if periods[axis] is None:
            return None, None
        if symbols[axis] != "two_pi":
            raise NotImplementedError(
                f"Native period derivation does not support {symbols[axis]!r} chart periods."
            )
        try:
            rates.append(rational_coefficient(velocity[axis], role="linear chart action"))
        except (TypeError, ValueError) as error:
            raise NotImplementedError(
                "Native period derivation requires rational exact chart actions."
            ) from error
    denominator = 1
    for rate in rates:
        denominator = lcm(denominator, rate.denominator)
    integer_rates = tuple(
        abs(rate.numerator * (denominator // rate.denominator)) for rate in rates
    )
    divisor = integer_rates[0]
    for rate in integer_rates[1:]:
        divisor = gcd(divisor, rate)
    turns = Fraction(denominator, divisor)
    try:
        period = float(tau * turns)
    except (OverflowError, ValueError) as error:
        raise NotImplementedError(
            "The exact native period has no finite binary64 realization."
        ) from error
    if not np.isfinite(period) or period <= 0.0:
        raise NotImplementedError(
            "The exact native period has no positive finite binary64 realization."
        )
    return period, "two_pi"


class ConicRootLinePCurve(AbstractCurve):
    """Exact conic-root chart line with an ordinary JAX numerical realization."""

    root: ConicParameterRoot
    phase_shift_exact: ExactScalar = eqx.field(static=True)
    height_origin_exact: ExactScalar = eqx.field(static=True)
    height_direction_exact: ExactScalar = eqx.field(static=True)
    angle: Array
    height_origin: Array
    height_direction: Array
    parameter_range: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        root: ConicParameterRoot,
        height_origin: float | ExactScalar,
        height_direction: float | ExactScalar,
        /,
        *,
        angle_realization: float | None = None,
        phase_shift_exact: ExactScalar = Fraction(0),
        parameter_range: tuple[float, float] | None = (0.0, 1.0),
    ) -> None:
        if not isinstance(root, ConicParameterRoot):
            raise TypeError("ConicRootLinePCurve.root must be a ConicParameterRoot.")
        if not isinstance(phase_shift_exact, (Fraction, RadicalElement)):
            raise TypeError(
                "ConicRootLinePCurve.phase_shift_exact must be an ExactScalar."
            )
        domain = None
        if parameter_range is not None:
            first, last = map(float, parameter_range)
            if not np.isfinite(first) or not np.isfinite(last) or not first < last:
                raise ValueError(
                    "ConicRootLinePCurve.parameter_range must be finite and increasing."
                )
            domain = (first, last)

        def exact_coefficient(
            value: float | ExactScalar, name: str, /
        ) -> tuple[ExactScalar, float]:
            if isinstance(value, (Fraction, RadicalElement)):
                exact = value
            else:
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(f"ConicRootLinePCurve.{name} must be finite.")
                exact = Fraction.from_float(numeric)
            try:
                shadow = float(exact)
            except OverflowError as error:
                raise ValueError(
                    f"ConicRootLinePCurve.{name} must have a finite numeric shadow."
                ) from error
            if not np.isfinite(shadow):
                raise ValueError(
                    f"ConicRootLinePCurve.{name} must have a finite numeric shadow."
                )
            return exact, shadow

        origin_exact, origin_numeric = exact_coefficient(height_origin, "height_origin")
        direction_exact, direction_numeric = exact_coefficient(
            height_direction, "height_direction"
        )
        if direction_exact == 0 or direction_numeric == 0.0:
            raise ValueError(
                "ConicRootLinePCurve.height_direction must be nonzero and representable."
            )
        angle_lower, angle_upper = _conic_root_phase_enclosure(root, phase_shift_exact)
        try:
            phase_numeric = float(phase_shift_exact)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "ConicRootLinePCurve phase shift must have a finite numeric shadow."
            ) from error
        if not np.isfinite(phase_numeric):
            raise ValueError(
                "ConicRootLinePCurve phase shift must have a finite numeric shadow."
            )
        numeric_angle = (
            root.parameter + phase_numeric
            if angle_realization is None
            else float(angle_realization)
        )
        if (
            not np.isfinite(numeric_angle)
            or not angle_lower <= numeric_angle <= angle_upper
        ):
            raise ValueError(
                "ConicRootLinePCurve angle realization must lie in its shifted root enclosure."
            )
        self.root = root
        self.phase_shift_exact = phase_shift_exact
        self.height_origin_exact = origin_exact
        self.height_direction_exact = direction_exact
        self.angle = jnp.asarray(numeric_angle, dtype=jnp.float64)
        self.height_origin = jnp.asarray(origin_numeric, dtype=jnp.float64)
        self.height_direction = jnp.asarray(direction_numeric, dtype=jnp.float64)
        self.parameter_range = domain

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float] | None:
        return self.parameter_range

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.angle.dtype)
        angle = jnp.broadcast_to(self.angle, values.shape)
        height = self.height_origin + self.height_direction * values
        return jnp.stack((angle, height), axis=-1)

    def is_c1_on(self, first: float, last: float, /) -> bool:
        AbstractCurve.is_c1_on(self, first, last)
        return True

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        self.is_c1_on(first, last)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Curve derivative bounds support order=1 only.")
        from .._rational import fraction_down, fraction_up

        # The angle is the constant root phase; only the height coordinate moves.
        lower, upper = coefficient_interval(self.height_direction_exact, bits=128)
        return (
            np.asarray((0.0, fraction_down(lower)), dtype=np.float64),
            np.asarray((0.0, fraction_up(upper)), dtype=np.float64),
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._rational import fraction_down, fraction_up

        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("ConicRootLinePCurve bounds must be finite and ordered.")
        # The exact affine source extension encloses isolating boxes across a
        # proved trim endpoint; the authored parameter range remains unchanged.
        parameters = tuple(Fraction.from_float(float(value)) for value in (first, last))
        heights = tuple(
            self.height_origin_exact + self.height_direction_exact * value
            for value in parameters
        )
        height_lower = coefficient_interval(min(heights), bits=128)[0]
        height_upper = coefficient_interval(max(heights), bits=128)[1]
        angle_lower, angle_upper = _conic_root_phase_enclosure(
            self.root, self.phase_shift_exact
        )
        return np.asarray(
            (
                (angle_lower, fraction_down(height_lower)),
                (angle_upper, fraction_up(height_upper)),
            ),
            dtype=np.float64,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": "conic-root-line-pcurve",
            "root": self.root.identity_payload(),
            "phase_shift": coefficient_payload(self.phase_shift_exact),
            "height_origin": coefficient_payload(self.height_origin_exact),
            "height_direction": coefficient_payload(self.height_direction_exact),
            "parameter_range": (
                None
                if self.parameter_range is None
                else tuple(float(value).hex() for value in self.parameter_range)
            ),
        }


class PlaneCylinderSectionPCurve(AbstractCurve):
    """Exact affine-harmonic UV lift of a plane/cylinder conic section."""

    offset_exact: tuple[ExactScalar, ExactScalar] = eqx.field(static=True)
    linear_exact: tuple[ExactScalar, ExactScalar] = eqx.field(static=True)
    cosine_exact: tuple[ExactScalar, ExactScalar] = eqx.field(static=True)
    sine_exact: tuple[ExactScalar, ExactScalar] = eqx.field(static=True)
    angle_offset_root: ConicParameterRoot | None
    parameter_range: tuple[float, float] = eqx.field(static=True)
    offset: Array
    linear: Array
    cosine: Array
    sine: Array
    angle_offset: Array

    def __init__(
        self,
        offset_exact: tuple[ExactScalar, ExactScalar],
        linear_exact: tuple[ExactScalar, ExactScalar],
        cosine_exact: tuple[ExactScalar, ExactScalar],
        sine_exact: tuple[ExactScalar, ExactScalar],
        /,
        *,
        parameter_range: tuple[float, float],
        angle_offset_root: ConicParameterRoot | None = None,
    ) -> None:
        from math import isfinite

        def exact_vector(
            values: tuple[ExactScalar, ExactScalar], name: str, /
        ) -> tuple[tuple[ExactScalar, ExactScalar], np.ndarray]:
            if (
                not isinstance(values, tuple)
                or len(values) != 2
                or any(
                    not isinstance(value, (Fraction, RadicalElement)) for value in values
                )
            ):
                raise TypeError(
                    f"PlaneCylinderSectionPCurve.{name} must contain two exact source scalars."
                )
            try:
                shadow = np.asarray(
                    tuple(float(value) for value in values), dtype=np.float64
                )
            except (OverflowError, ValueError) as error:
                raise ValueError(
                    f"PlaneCylinderSectionPCurve.{name} must have finite numeric shadows."
                ) from error
            if not np.all(np.isfinite(shadow)):
                raise ValueError(
                    f"PlaneCylinderSectionPCurve.{name} must have finite numeric shadows."
                )
            return values, shadow

        first, last = map(float, parameter_range)
        if not isfinite(first) or not isfinite(last) or not first < last:
            raise ValueError(
                "PlaneCylinderSectionPCurve.parameter_range must be finite and increasing."
            )
        if angle_offset_root is not None:
            if not isinstance(angle_offset_root, ConicParameterRoot):
                raise TypeError(
                    "PlaneCylinderSectionPCurve.angle_offset_root must be a ConicParameterRoot or None."
                )
            phase_conic = angle_offset_root.conic
            if (
                not isinstance(phase_conic, CircleCurve)
                or phase_conic.ambient_dimension != 2
                or phase_conic.center_exact != (Fraction(0), Fraction(0))
                or phase_conic.first_axis_exact != (Fraction(1), Fraction(0))
                or phase_conic.second_axis_exact != (Fraction(0), Fraction(1))
                or phase_conic.radius_exact != 1
                or phase_conic.native_period_symbol() != "two_pi"
            ):
                raise ValueError(
                    "PlaneCylinderSectionPCurve phase roots must use the canonical unit-circle two_pi authority."
                )

        offset_source, offset_shadow = exact_vector(offset_exact, "offset_exact")
        linear_source, linear_shadow = exact_vector(linear_exact, "linear_exact")
        cosine_source, cosine_shadow = exact_vector(cosine_exact, "cosine_exact")
        sine_source, sine_shadow = exact_vector(sine_exact, "sine_exact")
        phase = 0.0 if angle_offset_root is None else angle_offset_root.parameter
        if not isfinite(phase):
            raise ValueError(
                "PlaneCylinderSectionPCurve phase root must have a finite numeric realization."
            )
        self.offset_exact = offset_source
        self.linear_exact = linear_source
        self.cosine_exact = cosine_source
        self.sine_exact = sine_source
        self.angle_offset_root = angle_offset_root
        self.parameter_range = (first, last)
        self.offset = jnp.asarray(offset_shadow, dtype=jnp.float64)
        self.linear = jnp.asarray(linear_shadow, dtype=jnp.float64)
        self.cosine = jnp.asarray(cosine_shadow, dtype=jnp.float64)
        self.sine = jnp.asarray(sine_shadow, dtype=jnp.float64)
        self.angle_offset = jnp.asarray(phase, dtype=jnp.float64)

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.parameter_range

    @property
    def period(self) -> float | None:
        return tau if self.linear_exact == (Fraction(0), Fraction(0)) else None

    def native_period_symbol(self, /) -> str | None:
        return "two_pi" if self.period is not None else None

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        harmonics = (
            self.offset
            + values[..., None] * self.linear
            + jnp.cos(values)[..., None] * self.cosine
            + jnp.sin(values)[..., None] * self.sine
        )
        return harmonics.at[..., 0].add(self.angle_offset)

    def is_c1_on(self, first: float, last: float, /) -> bool:
        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("Curve ranges must be finite and nondecreasing.")
        if first < self.parameter_range[0] or last > self.parameter_range[1]:
            raise ValueError("Curve range must lie in its source parameter domain.")
        return True

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
        )
        from .._rational import fraction_down, fraction_up

        self.is_c1_on(first, last)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Curve derivative bounds support order=1 only.")
        parameter = (np.asarray(first), np.asarray(last))
        cosine = interval_cos(parameter)
        sine = interval_sin(parameter)

        def exact_interval(value: ExactScalar, /) -> tuple[np.ndarray, np.ndarray]:
            lower, upper = coefficient_interval(value, bits=128)
            return np.asarray(fraction_down(lower)), np.asarray(fraction_up(upper))

        lower = np.empty((2,), dtype=np.float64)
        upper = np.empty((2,), dtype=np.float64)
        for axis in range(2):
            derivative = exact_interval(self.linear_exact[axis])
            derivative = interval_add(
                derivative,
                interval_multiply(exact_interval(-self.cosine_exact[axis]), sine),
            )
            derivative = interval_add(
                derivative,
                interval_multiply(exact_interval(self.sine_exact[axis]), cosine),
            )
            lower[axis], upper[axis] = float(derivative[0]), float(derivative[1])
        return lower, upper

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
        )
        from .._rational import fraction_down, fraction_up

        self.is_c1_on(first, last)
        parameter = (np.asarray(first), np.asarray(last))
        cosine = interval_cos(parameter)
        sine = interval_sin(parameter)

        def exact_interval(value: ExactScalar, /) -> tuple[np.ndarray, np.ndarray]:
            lower, upper = coefficient_interval(value, bits=128)
            return np.asarray(fraction_down(lower)), np.asarray(fraction_up(upper))

        lower = np.empty((2,), dtype=np.float64)
        upper = np.empty((2,), dtype=np.float64)
        for axis in range(2):
            value = exact_interval(self.offset_exact[axis])
            value = interval_add(
                value,
                interval_multiply(exact_interval(self.linear_exact[axis]), parameter),
            )
            value = interval_add(
                value,
                interval_multiply(exact_interval(self.cosine_exact[axis]), cosine),
            )
            value = interval_add(
                value,
                interval_multiply(exact_interval(self.sine_exact[axis]), sine),
            )
            if axis == 0 and self.angle_offset_root is not None:
                value = interval_add(
                    value,
                    (
                        np.asarray(self.angle_offset_root.parameter_lower),
                        np.asarray(self.angle_offset_root.parameter_upper),
                    ),
                )
            lower[axis], upper[axis] = float(value[0]), float(value[1])
        return np.stack((lower, upper))

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": "plane-cylinder-section-pcurve",
            "offset": tuple(coefficient_payload(value) for value in self.offset_exact),
            "linear": tuple(coefficient_payload(value) for value in self.linear_exact),
            "cosine": tuple(coefficient_payload(value) for value in self.cosine_exact),
            "sine": tuple(coefficient_payload(value) for value in self.sine_exact),
            "angle_offset_root": None
            if self.angle_offset_root is None
            else self.angle_offset_root.identity_payload(),
            "parameter_range": tuple(
                float(value).hex() for value in self.parameter_range
            ),
        }


class PlanePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array

    def __init__(
        self,
        origin: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        /,
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        # Equal-rank operands: batched charts must not rely on rank promotion.
        shape = (1,) * (parameters_.ndim - 1) + (self.origin.shape[-1],)
        return (
            self.origin.reshape(shape)
            + parameters_[..., :1] * self.first_axis.reshape(shape)
            + parameters_[..., 1:2] * self.second_axis.reshape(shape)
        )

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameter_box(parameter_box)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Plane derivative bounds support order=1 only.")
        jacobian = np.column_stack(
            (np.asarray(self.first_axis), np.asarray(self.second_axis))
        )
        return jacobian, jacobian


class CylinderPatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array
    origin_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    first_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    second_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    radius_exact: ExactScalar = eqx.field(static=True)

    def __init__(
        self,
        origin: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        radius: ConvertibleToArray,
        /,
        *,
        origin_exact: tuple[ExactScalar, ...] | None = None,
        first_axis_exact: tuple[ExactScalar, ...] | None = None,
        second_axis_exact: tuple[ExactScalar, ...] | None = None,
        axis_exact: tuple[ExactScalar, ...] | None = None,
        radius_exact: ExactScalar | None = None,
    ) -> None:
        numeric = tuple(
            np.asarray(value, dtype=np.float64).reshape((3,))
            for value in (origin, first_axis, second_axis, axis)
        )
        radius_numeric = float(np.asarray(radius, dtype=np.float64).reshape(()))
        if any(not np.all(np.isfinite(value)) for value in numeric) or not np.isfinite(
            radius_numeric
        ):
            raise ValueError("CylinderPatch numeric realization must be finite.")

        def exact_vector(
            values: tuple[ExactScalar, ...] | None,
            shadow: np.ndarray,
            name: str,
        ) -> tuple[ExactScalar, ...]:
            result = (
                tuple(Fraction.from_float(float(value)) for value in shadow)
                if values is None
                else tuple(values)
            )
            if len(result) != 3 or any(
                not isinstance(value, (Fraction, RadicalElement)) for value in result
            ):
                raise TypeError(
                    f"CylinderPatch.{name} must contain three ExactScalar values."
                )
            if not np.array_equal(
                np.asarray(tuple(float(value) for value in result), dtype=np.float64),
                shadow,
            ):
                raise ValueError(
                    f"CylinderPatch.{name} shadow must round its exact source."
                )
            return result

        exact_vectors = tuple(
            exact_vector(values, shadow, name)
            for values, shadow, name in zip(
                (origin_exact, first_axis_exact, second_axis_exact, axis_exact),
                numeric,
                ("origin_exact", "first_axis_exact", "second_axis_exact", "axis_exact"),
                strict=True,
            )
        )
        exact_radius = (
            Fraction.from_float(radius_numeric) if radius_exact is None else radius_exact
        )
        if (
            not isinstance(exact_radius, (Fraction, RadicalElement))
            or float(exact_radius) != radius_numeric
        ):
            raise ValueError("CylinderPatch radius shadow must round its exact source.")
        self.origin, self.first_axis, self.second_axis, self.axis = tuple(
            jnp.asarray(value, dtype=jnp.float64) for value in numeric
        )
        self.radius = jnp.asarray(radius_numeric, dtype=jnp.float64)
        (
            self.origin_exact,
            self.first_axis_exact,
            self.second_axis_exact,
            self.axis_exact,
        ) = exact_vectors
        self.radius_exact = exact_radius

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "exact-cylinder-definition",
            "origin": tuple(coefficient_payload(value) for value in self.origin_exact),
            "first_axis": tuple(
                coefficient_payload(value) for value in self.first_axis_exact
            ),
            "second_axis": tuple(
                coefficient_payload(value) for value in self.second_axis_exact
            ),
            "axis": tuple(coefficient_payload(value) for value in self.axis_exact),
            "radius": coefficient_payload(self.radius_exact),
        }

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle = parameters_[..., 0]
        height = parameters_[..., 1]
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        return self.origin + self.radius * radial + height[..., None] * self.axis

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
        )
        from .._rational import fraction_down, fraction_up

        box = np.asarray(self.validate_parameter_box(parameter_box), dtype=np.float64)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Cylinder derivative bounds support order=1 only.")

        def exact_interval(value: ExactScalar, /) -> tuple[np.ndarray, np.ndarray]:
            lower, upper = coefficient_interval(value, bits=128)
            return (
                np.asarray(fraction_down(lower), dtype=np.float64),
                np.asarray(fraction_up(upper), dtype=np.float64),
            )

        def vector_interval(
            values: tuple[ExactScalar, ...], /
        ) -> tuple[np.ndarray, np.ndarray]:
            components = tuple(exact_interval(value) for value in values)
            return (
                np.asarray(tuple(float(value[0]) for value in components)),
                np.asarray(tuple(float(value[1]) for value in components)),
            )

        angle = (box[0, 0], box[1, 0])
        sine, cosine = interval_sin(angle), interval_cos(angle)
        radius = exact_interval(self.radius_exact)
        negative_radius = (-radius[1], -radius[0])
        first_axis = vector_interval(self.first_axis_exact)
        second_axis = vector_interval(self.second_axis_exact)
        axis = vector_interval(self.axis_exact)
        first_term = interval_multiply(
            interval_multiply(negative_radius, sine), first_axis
        )
        second_term = interval_multiply(interval_multiply(radius, cosine), second_axis)
        lower, upper = interval_add(first_term, second_term)
        return (
            np.column_stack((lower, axis[0])),
            np.column_stack((upper, axis[1])),
        )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return tau, None

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return "two_pi", None


class ConePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    reference_radius: Array
    semi_angle: Array

    def __init__(
        self,
        origin: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        reference_radius: ConvertibleToArray,
        semi_angle: ConvertibleToArray,
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=jnp.float64).reshape((3,))
        self.reference_radius = jnp.asarray(reference_radius, dtype=jnp.float64).reshape(
            ()
        )
        self.semi_angle = jnp.asarray(semi_angle, dtype=jnp.float64).reshape(())

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle = parameters_[..., 0]
        axial = parameters_[..., 1]
        radius = self.reference_radius + axial * jnp.tan(self.semi_angle)
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        return self.origin + axial[..., None] * self.axis + radius[..., None] * radial

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_divide,
            interval_multiply,
            interval_sin,
        )

        box = np.asarray(self.validate_parameter_box(parameter_box), dtype=np.float64)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Cone derivative bounds support order=1 only.")
        angle = (box[0, 0], box[1, 0])
        sine, cosine = interval_sin(angle), interval_cos(angle)
        first_axis = np.asarray(self.first_axis, dtype=np.float64)
        second_axis = np.asarray(self.second_axis, dtype=np.float64)
        radial = interval_add(
            interval_multiply(cosine, (first_axis, first_axis)),
            interval_multiply(sine, (second_axis, second_axis)),
        )
        tangent = interval_add(
            interval_multiply(
                (-np.asarray(sine[1]), -np.asarray(sine[0])), (first_axis, first_axis)
            ),
            interval_multiply(cosine, (second_axis, second_axis)),
        )
        semi_angle = np.float64(np.asarray(self.semi_angle))
        tan_angle = interval_divide(
            interval_sin((semi_angle, semi_angle)),
            interval_cos((semi_angle, semi_angle)),
        )
        axial = (box[0, 1], box[1, 1])
        reference_radius = np.float64(np.asarray(self.reference_radius))
        radius = interval_add(
            (reference_radius, reference_radius),
            interval_multiply(axial, tan_angle),
        )
        du = interval_multiply(radius, tangent)
        dv = interval_add(
            (np.asarray(self.axis), np.asarray(self.axis)),
            interval_multiply(tan_angle, radial),
        )
        return (
            np.column_stack((du[0], dv[0])),
            np.column_stack((du[1], dv[1])),
        )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return tau, None

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return "two_pi", None


class SpherePatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        radius: ConvertibleToArray,
    ) -> None:
        self.center = jnp.asarray(center, dtype=jnp.float64).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=jnp.float64).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=jnp.float64).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=jnp.float64).reshape((3,))
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude = parameters_[..., 0]
        latitude = parameters_[..., 1]
        equatorial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        direction = (
            jnp.cos(latitude)[..., None] * equatorial
            + jnp.sin(latitude)[..., None] * self.axis
        )
        return self.center + self.radius * direction

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
        )

        box = np.asarray(self.validate_parameter_box(parameter_box), dtype=np.float64)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Sphere derivative bounds support order=1 only.")
        longitude = (box[0, 0], box[1, 0])
        latitude = (box[0, 1], box[1, 1])
        sin_u, cos_u = interval_sin(longitude), interval_cos(longitude)
        sin_v, cos_v = interval_sin(latitude), interval_cos(latitude)
        first_axis = np.asarray(self.first_axis, dtype=np.float64)
        second_axis = np.asarray(self.second_axis, dtype=np.float64)
        axis = np.asarray(self.axis, dtype=np.float64)
        radial = interval_add(
            interval_multiply(cos_u, (first_axis, first_axis)),
            interval_multiply(sin_u, (second_axis, second_axis)),
        )
        tangent_u = interval_add(
            interval_multiply(
                (-np.asarray(sin_u[1]), -np.asarray(sin_u[0])), (first_axis, first_axis)
            ),
            interval_multiply(cos_u, (second_axis, second_axis)),
        )
        tangent_v = interval_add(
            interval_multiply((-np.asarray(sin_v[1]), -np.asarray(sin_v[0])), radial),
            interval_multiply(cos_v, (axis, axis)),
        )
        radius = np.float64(np.asarray(self.radius))
        du = interval_multiply((radius, radius), interval_multiply(cos_v, tangent_u))
        dv = interval_multiply((radius, radius), tangent_v)
        return (
            np.column_stack((du[0], dv[0])),
            np.column_stack((du[1], dv[1])),
        )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return tau, None

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return "two_pi", None


class TorusPatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    major_radius: Array
    minor_radius: Array
    center_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    first_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    second_axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    axis_exact: tuple[ExactScalar, ...] = eqx.field(static=True)
    major_radius_exact: ExactScalar = eqx.field(static=True)
    minor_radius_exact: ExactScalar = eqx.field(static=True)

    def __init__(
        self,
        center: ConvertibleToArray,
        first_axis: ConvertibleToArray,
        second_axis: ConvertibleToArray,
        axis: ConvertibleToArray,
        major_radius: ConvertibleToArray,
        minor_radius: ConvertibleToArray,
        *,
        center_exact: tuple[ExactScalar, ...] | None = None,
        first_axis_exact: tuple[ExactScalar, ...] | None = None,
        second_axis_exact: tuple[ExactScalar, ...] | None = None,
        axis_exact: tuple[ExactScalar, ...] | None = None,
        major_radius_exact: ExactScalar | None = None,
        minor_radius_exact: ExactScalar | None = None,
    ) -> None:
        numeric = tuple(
            np.asarray(value, dtype=np.float64).reshape((3,))
            for value in (center, first_axis, second_axis, axis)
        )
        radius_numeric = tuple(
            float(np.asarray(value, dtype=np.float64).reshape(()))
            for value in (major_radius, minor_radius)
        )
        if any(not np.all(np.isfinite(value)) for value in numeric) or not all(
            np.isfinite(value) for value in radius_numeric
        ):
            raise ValueError("TorusPatch numeric realization must be finite.")

        def exact_vector(
            values: tuple[ExactScalar, ...] | None,
            shadow: np.ndarray,
            name: str,
        ) -> tuple[ExactScalar, ...]:
            result = (
                tuple(Fraction.from_float(float(value)) for value in shadow)
                if values is None
                else tuple(values)
            )
            if len(result) != 3:
                raise ValueError(f"TorusPatch.{name} must contain three coordinates.")
            if any(not isinstance(value, (Fraction, RadicalElement)) for value in result):
                raise TypeError(
                    f"TorusPatch.{name} must contain three ExactScalar values."
                )
            try:
                rounded = np.asarray(
                    tuple(float(value) for value in result), dtype=np.float64
                )
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    f"TorusPatch.{name} must have a finite binary64 realization."
                ) from error
            if not np.all(np.isfinite(rounded)) or not np.array_equal(rounded, shadow):
                raise ValueError(f"TorusPatch.{name} shadow must round its exact source.")
            return result

        def exact_scalar(
            value: ExactScalar | None,
            shadow: float,
            name: str,
        ) -> ExactScalar:
            result = Fraction.from_float(shadow) if value is None else value
            if not isinstance(result, (Fraction, RadicalElement)):
                raise TypeError(f"TorusPatch.{name} must be an ExactScalar.")
            try:
                rounded = float(result)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    f"TorusPatch.{name} must have a finite binary64 realization."
                ) from error
            if not np.isfinite(rounded) or rounded != shadow:
                raise ValueError(f"TorusPatch.{name} shadow must round its exact source.")
            return result

        exact_vectors = tuple(
            exact_vector(values, shadow, name)
            for values, shadow, name in zip(
                (center_exact, first_axis_exact, second_axis_exact, axis_exact),
                numeric,
                ("center_exact", "first_axis_exact", "second_axis_exact", "axis_exact"),
                strict=True,
            )
        )
        exact_major = exact_scalar(
            major_radius_exact, radius_numeric[0], "major_radius_exact"
        )
        exact_minor = exact_scalar(
            minor_radius_exact, radius_numeric[1], "minor_radius_exact"
        )
        self.center, self.first_axis, self.second_axis, self.axis = tuple(
            jnp.asarray(value, dtype=jnp.float64) for value in numeric
        )
        self.major_radius, self.minor_radius = tuple(
            jnp.asarray(value, dtype=jnp.float64) for value in radius_numeric
        )
        (
            self.center_exact,
            self.first_axis_exact,
            self.second_axis_exact,
            self.axis_exact,
        ) = exact_vectors
        self.major_radius_exact = exact_major
        self.minor_radius_exact = exact_minor

    def identity_payload(self, /) -> dict[str, object]:
        return {
            "kind": "exact-torus-definition",
            "center": tuple(coefficient_payload(value) for value in self.center_exact),
            "first_axis": tuple(
                coefficient_payload(value) for value in self.first_axis_exact
            ),
            "second_axis": tuple(
                coefficient_payload(value) for value in self.second_axis_exact
            ),
            "axis": tuple(coefficient_payload(value) for value in self.axis_exact),
            "major_radius": coefficient_payload(self.major_radius_exact),
            "minor_radius": coefficient_payload(self.minor_radius_exact),
        }

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude = parameters_[..., 0]
        tube_angle = parameters_[..., 1]
        radial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        ring_radius = self.major_radius + self.minor_radius * jnp.cos(tube_angle)
        return (
            self.center
            + ring_radius[..., None] * radial
            + self.minor_radius * jnp.sin(tube_angle)[..., None] * self.axis
        )

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import (
            interval_add,
            interval_cos,
            interval_multiply,
            interval_sin,
        )
        from .._rational import fraction_down, fraction_up

        box = np.asarray(self.validate_parameter_box(parameter_box), dtype=np.float64)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Torus derivative bounds support order=1 only.")

        def exact_interval(value: ExactScalar, /) -> tuple[np.ndarray, np.ndarray]:
            lower, upper = coefficient_interval(value, bits=128)
            return (
                np.asarray(fraction_down(lower), dtype=np.float64),
                np.asarray(fraction_up(upper), dtype=np.float64),
            )

        def vector_interval(
            values: tuple[ExactScalar, ...], /
        ) -> tuple[np.ndarray, np.ndarray]:
            components = tuple(exact_interval(value) for value in values)
            return (
                np.asarray(tuple(float(value[0]) for value in components)),
                np.asarray(tuple(float(value[1]) for value in components)),
            )

        longitude = (box[0, 0], box[1, 0])
        tube = (box[0, 1], box[1, 1])
        sin_u, cos_u = interval_sin(longitude), interval_cos(longitude)
        sin_v, cos_v = interval_sin(tube), interval_cos(tube)
        first_axis = vector_interval(self.first_axis_exact)
        second_axis = vector_interval(self.second_axis_exact)
        axis = vector_interval(self.axis_exact)
        radial = interval_add(
            interval_multiply(cos_u, first_axis),
            interval_multiply(sin_u, second_axis),
        )
        tangent_u = interval_add(
            interval_multiply((-np.asarray(sin_u[1]), -np.asarray(sin_u[0])), first_axis),
            interval_multiply(cos_u, second_axis),
        )
        major = exact_interval(self.major_radius_exact)
        minor = exact_interval(self.minor_radius_exact)
        ring_radius = interval_add(major, interval_multiply(minor, cos_v))
        du = interval_multiply(ring_radius, tangent_u)
        tube_tangent = interval_add(
            interval_multiply((-np.asarray(sin_v[1]), -np.asarray(sin_v[0])), radial),
            interval_multiply(cos_v, axis),
        )
        dv = interval_multiply(minor, tube_tangent)
        return (
            np.column_stack((du[0], dv[0])),
            np.column_stack((du[1], dv[1])),
        )

    @property
    def periods(self) -> tuple[float | None, float | None]:
        return tau, tau

    def native_period_symbols(self, /) -> tuple[str | None, str | None]:
        return "two_pi", "two_pi"


type _BinarySourceBank = float | list[_BinarySourceBank] | tuple[_BinarySourceBank, ...]
type _ExactSourceBank = ExactScalar | tuple[_ExactSourceBank, ...]


def _exact_bspline_source_bank(
    values: object | None, represented: Array, name: str, /
) -> object | None:
    """Return the exact bank given or derivable from concrete numeric leaves.

    Traced leaves are never converted on the host: without an authored bank they
    carry none, and an authored bank is checked against their static shape only.
    """
    traced = isinstance(represented, Tracer)
    if values is None and traced:
        return None
    numeric = None if traced else np.asarray(represented, dtype=np.float64)

    def freeze(value: object) -> object:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (tuple, list)):
            return tuple(freeze(component) for component in value)
        return value

    def binary_bank(value: NDArray[np.float64] | _BinarySourceBank) -> _ExactSourceBank:
        if isinstance(value, np.ndarray):
            return binary_bank(value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(binary_bank(component) for component in value)
        return Fraction.from_float(float(value))

    bank = (
        binary_bank(cast(NDArray[np.float64], numeric))
        if values is None
        else freeze(values)
    )

    def shape(value: object) -> tuple[int, ...]:
        if not isinstance(value, tuple):
            return ()
        if not value:
            return (0,)
        child_shapes = tuple(shape(component) for component in value)
        if any(item != child_shapes[0] for item in child_shapes[1:]):
            raise ValueError(f"{name} must have a rectangular exact source bank.")
        return (len(value), *child_shapes[0])

    if shape(bank) != tuple(represented.shape):
        raise ValueError(f"{name} must match its numeric source-bank shape.")

    def leaves(value: object) -> tuple[ExactScalar, ...]:
        if isinstance(value, tuple):
            return tuple(component for item in value for component in leaves(item))
        if not isinstance(value, (Fraction, RadicalElement)):
            raise TypeError(f"{name} must contain exact rational or radical scalars.")
        return (value,)

    exact_values = leaves(bank)
    if numeric is None:
        return bank
    for value, shadow in zip(exact_values, numeric.reshape((-1,)), strict=True):
        try:
            rounded = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must have a finite binary64 realization."
            ) from error
        if not np.isfinite(rounded) or rounded != float(shadow):
            raise ValueError(f"{name} does not match its numeric realization.")
    return bank


def _stored_exact_bank(bank: object | None, represented: Array, name: str, /) -> object:
    """Return a stored exact bank, deriving it from concrete leaves on demand."""
    if bank is not None:
        return bank
    derived = _exact_bspline_source_bank(None, represented, name)
    if derived is None:
        raise TypeError(
            f"{name} is unavailable for traced B-spline coefficients; exact source "
            "operations are host-only."
        )
    return derived


def _validated_spline_leaf(
    value: Array,
    invalid: Callable[[Array], Array],
    message: str,
    /,
) -> Array:
    """Refuse an invalid concrete leaf eagerly and a traced leaf at execution.

    ``invalid`` uses array operators only, so it applies to host and traced leaves.
    """
    if isinstance(value, Tracer):
        return eqx.error_if(value, invalid(value), message)
    if bool(invalid(cast(Array, np.asarray(value)))):
        raise ValueError(message)
    return value


def _nonfinite(value: Array, /) -> Array:
    return (~(abs(value) < np.inf)).any()


def _nonpositive(value: Array, /) -> Array:
    return (value <= 0.0).any()


def _decreasing(value: Array, /) -> Array:
    return (value[1:] < value[:-1]).any()


def _empty_domain(degree: int, count: int, /) -> Callable[[Array], Array]:
    return lambda knots: knots[degree] >= knots[count]


class BSplineSurfacePatch(AbstractSurfacePatch):
    """Tensor-product rational B-spline surface with expanded knot vectors."""

    control_points: Array
    weights: Array
    u_knots: Array
    v_knots: Array
    u_degree: int = eqx.field(static=True)
    v_degree: int = eqx.field(static=True)
    # Exact source banks: authored or derived from concrete leaves; ``None`` only
    # for traced leaves without an authored bank (see the ``exact_*`` properties).
    exact_control_point_bank: tuple[tuple[tuple[ExactScalar, ...], ...], ...] | None = (
        eqx.field(static=True)
    )
    exact_weight_bank: tuple[tuple[ExactScalar, ...], ...] | None = eqx.field(static=True)
    exact_u_knot_bank: tuple[Fraction, ...] | None = eqx.field(static=True)
    exact_v_knot_bank: tuple[Fraction, ...] | None = eqx.field(static=True)
    control_point_ids: tuple[tuple[int, ...], ...] | None = eqx.field(static=True)

    def __init__(
        self,
        control_points: ConvertibleToArray,
        weights: ConvertibleToArray,
        u_knots: ConvertibleToArray,
        v_knots: ConvertibleToArray,
        u_degree: int,
        v_degree: int,
        *,
        exact_control_points: tuple[tuple[tuple[ExactScalar, ...], ...], ...]
        | None = None,
        exact_weights: tuple[tuple[ExactScalar, ...], ...] | None = None,
        exact_u_knots: tuple[Fraction, ...] | None = None,
        exact_v_knots: tuple[Fraction, ...] | None = None,
        control_point_ids: ConvertibleToArray | None = None,
    ) -> None:
        control_points_ = jnp.asarray(control_points, dtype=jnp.float64)
        weights_ = jnp.asarray(weights, dtype=jnp.float64)
        u_knots_ = jnp.asarray(u_knots, dtype=jnp.float64).reshape((-1,))
        v_knots_ = jnp.asarray(v_knots, dtype=jnp.float64).reshape((-1,))
        if (
            isinstance(u_degree, bool)
            or not isinstance(u_degree, int)
            or isinstance(v_degree, bool)
            or not isinstance(v_degree, int)
        ):
            raise TypeError("B-spline degrees must be integers.")
        message = "B-spline surface coefficients must be finite."
        control_points_ = _validated_spline_leaf(control_points_, _nonfinite, message)
        weights_ = _validated_spline_leaf(weights_, _nonfinite, message)
        u_knots_ = _validated_spline_leaf(u_knots_, _nonfinite, message)
        v_knots_ = _validated_spline_leaf(v_knots_, _nonfinite, message)
        # Positive weights are the admissibility contract of rational CAD splines:
        # convex-hull enclosure proofs depend on them.
        weights_ = _validated_spline_leaf(
            weights_, _nonpositive, "B-spline surface weights must be positive."
        )
        message = "B-spline surface knots must be nondecreasing."
        u_knots_ = _validated_spline_leaf(u_knots_, _decreasing, message)
        v_knots_ = _validated_spline_leaf(v_knots_, _decreasing, message)
        if control_points_.ndim != 3 or control_points_.shape[-1] != 3:
            raise ValueError("control_points must have shape (num_u, num_v, 3).")
        if weights_.shape != control_points_.shape[:2]:
            raise ValueError("weights must match the control-point grid.")
        if int(u_degree) < 1 or int(v_degree) < 1:
            raise ValueError("B-spline degrees must be positive.")
        if control_points_.shape[0] <= int(u_degree) or control_points_.shape[1] <= int(
            v_degree
        ):
            raise ValueError("Each B-spline control-point axis must exceed its degree.")
        if u_knots_.shape[0] != control_points_.shape[0] + int(u_degree) + 1:
            raise ValueError(
                "u_knots length is inconsistent with control points and degree."
            )
        if v_knots_.shape[0] != control_points_.shape[1] + int(v_degree) + 1:
            raise ValueError(
                "v_knots length is inconsistent with control points and degree."
            )
        message = "B-spline surface domains must be nonempty."
        u_knots_ = _validated_spline_leaf(
            u_knots_, _empty_domain(u_degree, control_points_.shape[0]), message
        )
        v_knots_ = _validated_spline_leaf(
            v_knots_, _empty_domain(v_degree, control_points_.shape[1]), message
        )
        exact_control_points_ = cast(
            tuple[tuple[tuple[ExactScalar, ...], ...], ...] | None,
            _exact_bspline_source_bank(
                exact_control_points, control_points_, "exact_control_points"
            ),
        )
        exact_weights_ = cast(
            tuple[tuple[ExactScalar, ...], ...] | None,
            _exact_bspline_source_bank(exact_weights, weights_, "exact_weights"),
        )
        exact_u_knots_ = cast(
            tuple[Fraction, ...] | None,
            _exact_bspline_source_bank(exact_u_knots, u_knots_, "exact_u_knots"),
        )
        exact_v_knots_ = cast(
            tuple[Fraction, ...] | None,
            _exact_bspline_source_bank(exact_v_knots, v_knots_, "exact_v_knots"),
        )
        if any(
            not isinstance(value, Fraction)
            for knots in (exact_u_knots_, exact_v_knots_)
            if knots is not None
            for value in knots
        ):
            raise TypeError("Surface parameter knots must be exact Fractions.")
        if exact_weights_ is not None and any(
            value <= 0 for row in exact_weights_ for value in row
        ):
            raise ValueError("Exact B-spline surface weights must be positive.")
        for knots, name, degree, count in (
            (exact_u_knots_, "u_knots", u_degree, control_points_.shape[0]),
            (exact_v_knots_, "v_knots", v_degree, control_points_.shape[1]),
        ):
            if knots is None:
                continue
            if any(second < first for first, second in zip(knots, knots[1:])):
                raise ValueError(f"Exact {name} must be nondecreasing.")
            if any(
                first < second and float(first) == float(second)
                for first, second in zip(knots, knots[1:])
            ):
                raise ValueError(
                    "Exact surface knot intervals must remain distinct in binary64."
                )
            if knots[degree] >= knots[count]:
                raise ValueError("Exact B-spline surface domains must be nonempty.")
        self.control_points = control_points_
        self.weights = weights_
        self.u_knots = u_knots_
        self.v_knots = v_knots_
        self.u_degree = int(u_degree)
        self.v_degree = int(v_degree)
        self.exact_control_point_bank = exact_control_points_
        self.exact_weight_bank = exact_weights_
        self.exact_u_knot_bank = exact_u_knots_
        self.exact_v_knot_bank = exact_v_knots_
        control_point_ids_ = None
        if control_point_ids is not None:
            raw_ids = np.asarray(control_point_ids)
            if (
                raw_ids.shape != control_points_.shape[:2]
                or raw_ids.dtype.kind not in "iu"
            ):
                raise ValueError(
                    "control_point_ids must be an aligned integer control-grid bank."
                )
            if raw_ids.dtype.kind == "u" and np.any(raw_ids > np.iinfo(np.int64).max):
                raise ValueError(
                    "control_point_ids must fit the nonnegative int64 identity domain."
                )
            ids = np.asarray(raw_ids, dtype=np.int64)
            flat_ids = ids.reshape((-1,))
            if np.any(flat_ids < 0) or np.unique(flat_ids).size != flat_ids.size:
                raise ValueError(
                    "control_point_ids must be unique nonnegative source IDs."
                )
            control_point_ids_ = tuple(tuple(int(value) for value in row) for row in ids)
        self.control_point_ids = control_point_ids_

    @property
    def exact_control_points(self) -> tuple[tuple[tuple[ExactScalar, ...], ...], ...]:
        """Exact control points; host-only for traced leaves without a bank."""
        return cast(
            tuple[tuple[tuple[ExactScalar, ...], ...], ...],
            _stored_exact_bank(
                self.exact_control_point_bank,
                self.control_points,
                "exact_control_points",
            ),
        )

    @property
    def exact_weights(self) -> tuple[tuple[ExactScalar, ...], ...]:
        """Exact positive weights; host-only for traced leaves without a bank."""
        return cast(
            tuple[tuple[ExactScalar, ...], ...],
            _stored_exact_bank(self.exact_weight_bank, self.weights, "exact_weights"),
        )

    @property
    def exact_u_knots(self) -> tuple[Fraction, ...]:
        """Exact u knots; host-only for traced leaves without a bank."""
        return cast(
            tuple[Fraction, ...],
            _stored_exact_bank(self.exact_u_knot_bank, self.u_knots, "exact_u_knots"),
        )

    @property
    def exact_v_knots(self) -> tuple[Fraction, ...]:
        """Exact v knots; host-only for traced leaves without a bank."""
        return cast(
            tuple[Fraction, ...],
            _stored_exact_bank(self.exact_v_knot_bank, self.v_knots, "exact_v_knots"),
        )

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "rational-bspline-surface",
            "u_degree": self.u_degree,
            "v_degree": self.v_degree,
            "control_points": tuple(
                tuple(tuple(coefficient_payload(value) for value in row) for row in strip)
                for strip in self.exact_control_points
            ),
            "weights": tuple(
                tuple(coefficient_payload(value) for value in row)
                for row in self.exact_weights
            ),
            "u_knots": tuple(coefficient_payload(value) for value in self.exact_u_knots),
            "v_knots": tuple(coefficient_payload(value) for value in self.exact_v_knots),
            "control_point_ids": self.control_point_ids,
        }

    @property
    def parameter_domain(self, /) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (float(self.u_knots[self.u_degree]), float(self.v_knots[self.v_degree])),
            (
                float(self.u_knots[self.control_points.shape[0]]),
                float(self.v_knots[self.control_points.shape[1]]),
            ),
        )

    def is_c1_on(self, parameter_box: ConvertibleToArray, /) -> bool:
        box = np.asarray(self.validate_parameter_box(parameter_box), dtype=np.float64)
        for knots, degree, axis in (
            (np.asarray(self.u_knots), self.u_degree, 0),
            (np.asarray(self.v_knots), self.v_degree, 1),
        ):
            first, last = box[0, axis], box[1, axis]
            interior = knots[(knots > first) & (knots < last)]
            if interior.size and any(
                np.count_nonzero(knots == knot) >= degree for knot in np.unique(interior)
            ):
                return False
        return True

    def is_c1_along(
        self, fixed_axis: int, fixed_value: float, first: float, last: float, /
    ) -> bool:
        super().is_c1_along(fixed_axis, fixed_value, first, last)
        free_axis = 1 - fixed_axis
        knots, degree = (
            (np.asarray(self.u_knots), self.u_degree)
            if free_axis == 0
            else (np.asarray(self.v_knots), self.v_degree)
        )
        interior = knots[(knots > first) & (knots < last)]
        return not any(
            np.count_nonzero(knots == knot) >= degree for knot in np.unique(interior)
        )

    def prepare_spans(self, /) -> PreparedSourceSpans[RationalBSplineSurfaceSpan]:
        """Extract the exact rational Bernstein spans, with Hessian nets, of this content.

        Pure: every call extracts again. Consumers of long computations prepare
        once per carrier and pass the result to every routine that restricts
        spans (`bspline_surface_derivative_bounds`, source box enclosures).
        """
        from ._carrier_pieces import carrier_content_id

        return PreparedSourceSpans(
            carrier_content_id(self),
            rational_bspline_surface_spans(
                np.asarray(self.control_points, dtype=np.float64).tolist(),
                np.asarray(self.weights, dtype=np.float64).tolist(),
                np.asarray(self.u_knots, dtype=np.float64).tolist(),
                np.asarray(self.v_knots, dtype=np.float64).tolist(),
                self.u_degree,
                self.v_degree,
                exact_control_points=self.exact_control_points,
                exact_weights=self.exact_weights,
                exact_u_knots=self.exact_u_knots,
                exact_v_knots=self.exact_v_knots,
                include_hessian=True,
            ),
        )

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.control_points.dtype)
        arrays = (self.control_points, self.weights, self.u_knots, self.v_knots)
        degrees = (self.u_degree, self.v_degree)
        if values.ndim == 1:
            return _bspline_surface_rows(*arrays, values, *degrees)
        flat = values.reshape((-1, 2))
        return _bucketed_rows(
            lambda rows: _bspline_surface_rows(*arrays, rows, *degrees), flat, arrays
        ).reshape((*values.shape[:-1], 3))

    def derivative_bounds(
        self, parameter_box: ConvertibleToArray, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("B-spline surface derivative bounds support order=1 only.")
        return bspline_surface_derivative_bounds(
            self, self.prepare_spans(), parameter_box
        )


def bspline_surface_derivative_bounds(
    patch: BSplineSurfacePatch,
    spans: PreparedSourceSpans[RationalBSplineSurfaceSpan],
    parameter_box: ConvertibleToArray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the Jacobian columns of ``patch`` on a UV box from its prepared spans."""
    from .._rational import fraction_down, fraction_up

    box = np.asarray(patch.validate_parameter_box(parameter_box), dtype=np.float64)
    query = tuple(
        tuple(Fraction.from_float(float(value)) for value in box[:, axis])
        for axis in range(2)
    )
    component_bounds: list[list[list[tuple[Fraction, Fraction]]]] = [
        [[], []] for _ in range(3)
    ]
    for span in spans.spans:
        span_bounds = span.parameter_bounds
        restricted = tuple(
            (
                max(query[axis][0], span_bounds[axis][0]),
                min(query[axis][1], span_bounds[axis][1]),
            )
            for axis in range(2)
        )
        if any(lower > upper for lower, upper in restricted):
            continue
        enclosure = span.enclose(restricted)
        if not enclosure.positive_denominator:
            raise ValueError(
                "A rational B-spline span denominator is not strictly positive."
            )
        for component in range(3):
            component_bounds[component][0].append(enclosure.du_bounds[component])
            component_bounds[component][1].append(enclosure.dv_bounds[component])
    if any(
        not axis_bounds for component in component_bounds for axis_bounds in component
    ):
        raise ValueError(
            "The derivative query box does not intersect the active spline domain."
        )
    lower = np.empty((3, 2), dtype=np.float64)
    upper = np.empty((3, 2), dtype=np.float64)
    for component, axes in enumerate(component_bounds):
        for axis, intervals in enumerate(axes):
            lower[component, axis] = fraction_down(min(bound[0] for bound in intervals))
            upper[component, axis] = fraction_up(max(bound[1] for bound in intervals))
    return lower, upper


_QUERY_BUCKETS = (8, 64, 512, 4096)


def bucketed_query_rows(
    function: Callable[[Array], Array], rows: Array | np.ndarray, /
) -> np.ndarray:
    """Evaluate ``function`` on concrete query rows at a fixed set of batch shapes.

    Every distinct batch shape compiles its own executable (several megabytes
    for a spline jet program), and refinement queries vary in size every
    round. Rows are padded on the host with copies of the last row to the next
    bucket of `_QUERY_BUCKETS`; longer queries run in chunks of the largest
    bucket. A row-wise ``function`` therefore compiles at most one executable
    per bucket, independent of the query sizes a computation issues. Inputs
    reach the device by transfer, which compiles nothing.
    """
    host = np.asarray(rows)
    count = host.shape[0]
    if count == 0:
        return np.asarray(function(jax.device_put(host)))
    chunk = _QUERY_BUCKETS[-1]
    pieces = []
    for start in range(0, count, chunk):
        block = host[start : start + chunk]
        size = block.shape[0]
        bucket = next(value for value in _QUERY_BUCKETS if value >= size)
        if bucket != size:
            block = np.concatenate((block, np.repeat(block[-1:], bucket - size, axis=0)))
        pieces.append(np.asarray(function(jax.device_put(block)))[:size])
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def _bucketed_rows(
    function: Callable[[Array], Array],
    rows: Array,
    operands: tuple[Array, ...],
    /,
) -> Array:
    """Evaluate concrete query rows through `bucketed_query_rows`.

    Traced rows or operands keep their shape, so enclosing programs see no
    padding operations.
    """
    if rows.shape[0] == 0 or any(
        isinstance(value, Tracer) for value in (rows, *operands)
    ):
        return function(rows)
    return jax.device_put(bucketed_query_rows(function, rows))


def _bspline_surface_point(
    control_points: Array,
    weights: Array,
    u_knots: Array,
    v_knots: Array,
    parameters: Array,
    u_degree: int,
    v_degree: int,
    /,
) -> Array:
    u_stencil = bspline_jet_stencil(
        u_knots, parameters[0], degree=u_degree, maximum_order=0, bounds="clip"
    )
    v_stencil = bspline_jet_stencil(
        v_knots, parameters[1], degree=v_degree, maximum_order=0, bounds="clip"
    )
    plan = TensorBSplineJetPlan((u_stencil, v_stencil), maximum_order=0)
    return RationalSplineJet(plan, weights).apply(control_points)


@partial(jax.jit, static_argnums=(5, 6))
def _bspline_surface_rows(
    control_points: Array,
    weights: Array,
    u_knots: Array,
    v_knots: Array,
    parameters: Array,
    u_degree: int,
    v_degree: int,
    /,
) -> Array:
    """Compiled rational tensor B-spline evaluation keyed by degrees and shapes only.

    Surfaces of one structure share the traced program and its derivative and
    batching transforms; exact source banks never enter the compilation key.
    """
    arrays = (control_points, weights, u_knots, v_knots)
    if parameters.ndim == 1:
        return _bspline_surface_point(*arrays, parameters, u_degree, v_degree)
    return jax.vmap(
        lambda point: _bspline_surface_point(*arrays, point, u_degree, v_degree)
    )(parameters)


class BSplineCurve(AbstractCurve):
    """Rational B-spline curve used for exact parametric trim evaluation."""

    control_points: Array
    weights: Array
    knots: Array
    degree: int = eqx.field(static=True)
    # Exact source banks: authored or derived from concrete leaves; ``None`` only
    # for traced leaves without an authored bank (see the ``exact_*`` properties).
    exact_control_point_bank: tuple[tuple[ExactScalar, ...], ...] | None = eqx.field(
        static=True
    )
    exact_weight_bank: tuple[ExactScalar, ...] | None = eqx.field(static=True)
    exact_knot_bank: tuple[ExactScalar, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        control_points: ConvertibleToArray,
        weights: ConvertibleToArray,
        knots: ConvertibleToArray,
        degree: int,
        *,
        exact_control_points: tuple[tuple[ExactScalar, ...], ...] | None = None,
        exact_weights: tuple[ExactScalar, ...] | None = None,
        exact_knots: tuple[ExactScalar, ...] | None = None,
    ) -> None:
        control_points_ = jnp.asarray(control_points, dtype=jnp.float64)
        weights_ = jnp.asarray(weights, dtype=jnp.float64).reshape((-1,))
        knots_ = jnp.asarray(knots, dtype=jnp.float64).reshape((-1,))
        if (
            control_points_.ndim != 2
            or control_points_.shape[1] not in (2, 3)
            or control_points_.shape[0] != weights_.shape[0]
        ):
            raise ValueError(
                "Curve control points must have shape (num_control_points, 2 or 3) and align with weights."
            )
        message = "B-spline curve coefficients must be finite."
        control_points_ = _validated_spline_leaf(control_points_, _nonfinite, message)
        weights_ = _validated_spline_leaf(weights_, _nonfinite, message)
        knots_ = _validated_spline_leaf(knots_, _nonfinite, message)
        # Positive weights are the admissibility contract of rational CAD splines:
        # convex-hull enclosure proofs depend on them.
        weights_ = _validated_spline_leaf(
            weights_, _nonpositive, "B-spline curve weights must be positive."
        )
        knots_ = _validated_spline_leaf(
            knots_, _decreasing, "B-spline curve knots must be nondecreasing."
        )
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError("B-spline curve degree must be an integer.")
        if degree < 1:
            raise ValueError("B-spline curve degree must be positive.")
        if control_points_.shape[0] <= degree:
            raise ValueError("B-spline curve control-point count must exceed its degree.")
        if knots_.shape[0] != control_points_.shape[0] + degree + 1:
            raise ValueError("Curve knot vector length is inconsistent.")
        knots_ = _validated_spline_leaf(
            knots_,
            _empty_domain(degree, control_points_.shape[0]),
            "B-spline curve domain must be nonempty.",
        )
        exact_control_points_ = cast(
            tuple[tuple[ExactScalar, ...], ...] | None,
            _exact_bspline_source_bank(
                exact_control_points, control_points_, "exact_control_points"
            ),
        )
        exact_weights_ = cast(
            tuple[ExactScalar, ...] | None,
            _exact_bspline_source_bank(exact_weights, weights_, "exact_weights"),
        )
        exact_knots_ = cast(
            tuple[ExactScalar, ...] | None,
            _exact_bspline_source_bank(exact_knots, knots_, "exact_knots"),
        )
        if exact_weights_ is not None and any(value <= 0 for value in exact_weights_):
            raise ValueError("Exact B-spline curve weights must be positive.")
        if exact_knots_ is not None:
            if any(
                second < first for first, second in zip(exact_knots_, exact_knots_[1:])
            ):
                raise ValueError("Exact B-spline curve knots must be nondecreasing.")
            if any(
                first < second and float(first) == float(second)
                for first, second in zip(exact_knots_, exact_knots_[1:])
            ):
                raise ValueError(
                    "Exact B-spline knot intervals must remain distinct in binary64."
                )
            if exact_knots_[degree] >= exact_knots_[control_points_.shape[0]]:
                raise ValueError("Exact B-spline curve domain must be nonempty.")
        self.control_points = control_points_
        self.weights = weights_
        self.knots = knots_
        self.degree = int(degree)
        self.exact_control_point_bank = exact_control_points_
        self.exact_weight_bank = exact_weights_
        self.exact_knot_bank = exact_knots_

    @property
    def exact_control_points(self) -> tuple[tuple[ExactScalar, ...], ...]:
        """Exact control points; host-only for traced leaves without a bank."""
        return cast(
            tuple[tuple[ExactScalar, ...], ...],
            _stored_exact_bank(
                self.exact_control_point_bank,
                self.control_points,
                "exact_control_points",
            ),
        )

    @property
    def exact_weights(self) -> tuple[ExactScalar, ...]:
        """Exact positive weights; host-only for traced leaves without a bank."""
        return cast(
            tuple[ExactScalar, ...],
            _stored_exact_bank(self.exact_weight_bank, self.weights, "exact_weights"),
        )

    @property
    def exact_knots(self) -> tuple[ExactScalar, ...]:
        """Exact knots; host-only for traced leaves without a bank."""
        return cast(
            tuple[ExactScalar, ...],
            _stored_exact_bank(self.exact_knot_bank, self.knots, "exact_knots"),
        )

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "rational-bspline-curve",
            "degree": self.degree,
            "control_points": tuple(
                tuple(coefficient_payload(value) for value in row)
                for row in self.exact_control_points
            ),
            "weights": tuple(coefficient_payload(value) for value in self.exact_weights),
            "knots": tuple(coefficient_payload(value) for value in self.exact_knots),
        }

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return (
            float(self.knots[self.degree]),
            float(self.knots[self.control_points.shape[0]]),
        )

    def prepare_spans(self, /) -> PreparedSourceSpans[RationalBSplineCurveSpan]:
        """Extract the exact rational Bernstein spans of this source content.

        Pure: every call extracts again. Consumers of long computations prepare
        once per carrier and pass the result to `bspline_curve_bounds` and
        `bspline_curve_derivative_bounds`.
        """
        from ._carrier_pieces import carrier_content_id

        return PreparedSourceSpans(
            carrier_content_id(self),
            rational_bspline_curve_spans(
                np.asarray(self.control_points).tolist(),
                np.asarray(self.weights).tolist(),
                np.asarray(self.knots).tolist(),
                self.degree,
                exact_control_points=self.exact_control_points,
                exact_weights=self.exact_weights,
                exact_knots=self.exact_knots,
            ),
        )

    def is_c1_on(self, first: float, last: float, /) -> bool:
        if not super().is_c1_on(first, last):
            return False
        knots = np.asarray(self.knots)
        interior = knots[(knots > first) & (knots < last)]
        return not any(
            np.count_nonzero(knots == knot) >= self.degree for knot in np.unique(interior)
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        return bspline_curve_bounds(self, self.prepare_spans(), first, last)

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("B-spline curve derivative bounds support order=1 only.")
        return bspline_curve_derivative_bounds(self, self.prepare_spans(), first, last)

    @property
    def ambient_dimension(self) -> int:
        return self.control_points.shape[1]

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=self.control_points.dtype)
        arrays = (self.control_points, self.weights, self.knots)
        if values.ndim == 0:
            return _bspline_curve_rows(*arrays, values, self.degree)
        return _bucketed_rows(
            lambda rows: _bspline_curve_rows(*arrays, rows, self.degree),
            values.reshape((-1,)),
            arrays,
        ).reshape((*values.shape, self.control_points.shape[1]))


def bspline_curve_bounds(
    curve: BSplineCurve,
    spans: PreparedSourceSpans[RationalBSplineCurveSpan],
    first: float,
    last: float,
    /,
) -> np.ndarray:
    """Enclose ``curve`` on ``[first, last]`` from its prepared exact spans."""
    from .._rational import fraction_down, fraction_up

    if not np.isfinite(first) or not np.isfinite(last) or first > last:
        raise ValueError("BSplineCurve bounds must be finite and ordered.")
    lower = Fraction.from_float(float(first))
    upper = Fraction.from_float(float(last))
    component_bounds: list[list[tuple[Fraction, Fraction]]] = [
        [] for _ in range(curve.ambient_dimension)
    ]
    for span in spans.spans:
        span_lower, span_upper = span.parameter_bounds
        first_ = max(lower, span_lower)
        last_ = min(upper, span_upper)
        if first_ > last_:
            continue
        enclosure = span.enclose((first_, last_))
        for axis, value in enumerate(enclosure.value_bounds):
            component_bounds[axis].append(value)
    if any(not values for values in component_bounds):
        raise ValueError("BSplineCurve bounds do not intersect its active source domain.")
    return np.asarray(
        (
            tuple(
                fraction_down(min(value[0] for value in values))
                for values in component_bounds
            ),
            tuple(
                fraction_up(max(value[1] for value in values))
                for values in component_bounds
            ),
        ),
        dtype=np.float64,
    )


def bspline_curve_derivative_bounds(
    curve: BSplineCurve,
    spans: PreparedSourceSpans[RationalBSplineCurveSpan],
    first: float,
    last: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the first derivative of ``curve`` on ``[first, last]`` from its spans."""
    from .._rational import fraction_down, fraction_up

    if not curve.is_c1_on(first, last):
        raise NotImplementedError(
            "A derivative bound across a non-C1 B-spline knot is unsupported."
        )
    lower, upper = Fraction.from_float(float(first)), Fraction.from_float(float(last))
    component_bounds: list[list[tuple[Fraction, Fraction]]] = [
        [] for _ in range(curve.ambient_dimension)
    ]
    for span in spans.spans:
        span_lower, span_upper = span.parameter_bounds
        first_, last_ = max(lower, span_lower), min(upper, span_upper)
        if first_ > last_:
            continue
        enclosure = span.enclose((first_, last_))
        if not enclosure.positive_denominator:
            raise ValueError(
                "A rational B-spline curve denominator is not strictly positive."
            )
        for axis, value in enumerate(enclosure.first_derivative_bounds):
            component_bounds[axis].append(value)
    if any(not values for values in component_bounds):
        raise ValueError("Derivative bounds do not intersect the active B-spline domain.")
    return (
        np.asarray(
            tuple(
                fraction_down(min(value[0] for value in values))
                for values in component_bounds
            ),
            dtype=np.float64,
        ),
        np.asarray(
            tuple(
                fraction_up(max(value[1] for value in values))
                for values in component_bounds
            ),
            dtype=np.float64,
        ),
    )


def _bspline_curve_point(
    control_points: Array, weights: Array, knots: Array, parameter: Array, degree: int, /
) -> Array:
    stencil = bspline_jet_stencil(knots, parameter, degree=degree, maximum_order=0)
    plan = TensorBSplineJetPlan((stencil,), maximum_order=0)
    return RationalSplineJet(plan, weights).apply(control_points)


@partial(jax.jit, static_argnums=(4,))
def _bspline_curve_rows(
    control_points: Array, weights: Array, knots: Array, parameters: Array, degree: int, /
) -> Array:
    """Compiled rational B-spline curve evaluation keyed by degree and shapes only.

    Curves of one structure share the traced program and its derivative and
    batching transforms; exact source banks never enter the compilation key.
    """
    if parameters.ndim == 0:
        return _bspline_curve_point(control_points, weights, knots, parameters, degree)
    return jax.vmap(
        lambda parameter: _bspline_curve_point(
            control_points, weights, knots, parameter, degree
        )
    )(parameters)


type BRepCurve = AbstractCurve

type SurfacePatch = AbstractSurfacePatch


def surface_differential(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    parameters_ = jnp.asarray(parameters, dtype=jnp.float64)
    leading = parameters_.shape[:-1]
    values = jax.vmap(jax.jacfwd(lambda uv: patch.evaluate(uv)))(
        parameters_.reshape((-1, 2))
    )
    return values.reshape((*leading, 3, 2))


def surface_differential_rows(
    patch: AbstractSurfacePatch, parameters: np.ndarray, /
) -> np.ndarray:
    """Host method: `surface_differential` at concrete UV rows, bucketed shapes.

    Queries reach the compiled Jacobian programs only at the shapes of
    `bucketed_query_rows`, so a computation issuing many differently sized
    queries compiles a bounded set of executables.
    """
    host = np.asarray(parameters, dtype=np.float64)
    values = bucketed_query_rows(
        jax.vmap(jax.jacfwd(patch.evaluate)), host.reshape((-1, 2))
    )
    return values.reshape((*host.shape[:-1], 3, 2))


def surface_jacobian(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    return jnp.linalg.norm(
        jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
    )


def surface_normal(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    normal = jnp.cross(differential[..., :, 0], differential[..., :, 1])
    return normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)


def _require_exact_numeric_shadow(value: object, exact: object, role: str, /) -> None:
    if isinstance(value, Tracer):
        return
    exact_array = np.asarray(exact, dtype=object)
    expected = np.asarray(
        tuple(float(item) for item in exact_array.reshape((-1,))),
        dtype=np.float64,
    ).reshape(exact_array.shape)
    actual = np.asarray(value, dtype=np.float64)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"{role} execution shadow does not match its exact source.")


def _validate_placed_source_shadows(
    rotation: Array, translation: Array, affine: object, /
) -> None:
    from ._assembly import BRepAffinePlacement
    from ._placed import _exact_affine

    _exact_affine(rotation, translation, cast(BRepAffinePlacement | None, affine))


def _validate_conic_parameter_root(
    root: ConicParameterRoot, visited: set[int] | None = None, /
) -> None:
    if not isinstance(root, ConicParameterRoot):
        raise TypeError("root must be a ConicParameterRoot.")
    visited_ = set() if visited is None else visited
    identity = id(root)
    if identity in visited_:
        return
    visited_.add(identity)
    _validate_source_shadows(root.conic, visited_)
    expected = ConicParameterRoot(
        root.conic,
        root.cosine_exact,
        root.sine_exact,
        turns=root.turns,
    )
    if (
        root.parameter_lower != expected.parameter_lower
        or root.parameter_upper != expected.parameter_upper
        or root.root_id != expected.root_id
    ):
        raise ValueError(
            "ConicParameterRoot derived bounds or identity do not match its exact source."
        )


def _validate_source_shadows(
    value: AbstractCurve | AbstractSurfacePatch,
    visited: set[int],
    /,
) -> None:
    from ._boolean_core import (
        _pcurve_execution_shadows_match,
        AffinePCurve,
        PeriodicPCurve,
    )
    from ._intersection_curve import (
        CurveTrimSegment,
        IntersectionCurve,
        IntersectionPCurve,
    )
    from ._placed import PlacedCurve, PlacedSurface
    from ._tspline import TSplineSurfacePatch

    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)
    if isinstance(value, (AffinePCurve, PeriodicPCurve)) and not (
        _pcurve_execution_shadows_match(value)
    ):
        raise ValueError(
            "Boolean p-curve execution shadows do not match their exact source."
        )

    if isinstance(value, ExactLineCurve):
        _require_exact_numeric_shadow(
            value.first, value.first_exact, "ExactLineCurve.first"
        )
        _require_exact_numeric_shadow(value.last, value.last_exact, "ExactLineCurve.last")
    elif isinstance(value, LineCurve):
        _require_exact_numeric_shadow(
            value.origin, value.origin_exact, "LineCurve.origin"
        )
        _require_exact_numeric_shadow(
            value.direction, value.direction_exact, "LineCurve.direction"
        )
    elif isinstance(value, CircleCurve):
        _require_exact_numeric_shadow(
            value.center, value.center_exact, "CircleCurve.center"
        )
        _require_exact_numeric_shadow(
            value.first_axis, value.first_axis_exact, "CircleCurve.first_axis"
        )
        _require_exact_numeric_shadow(
            value.second_axis, value.second_axis_exact, "CircleCurve.second_axis"
        )
        _require_exact_numeric_shadow(
            value.radius, value.radius_exact, "CircleCurve.radius"
        )
    elif isinstance(value, EllipseCurve):
        for numeric, exact, role in (
            (value.center, value.center_exact, "EllipseCurve.center"),
            (value.first_axis, value.first_axis_exact, "EllipseCurve.first_axis"),
            (value.second_axis, value.second_axis_exact, "EllipseCurve.second_axis"),
            (value.first_radius, value.first_radius_exact, "EllipseCurve.first_radius"),
            (
                value.second_radius,
                value.second_radius_exact,
                "EllipseCurve.second_radius",
            ),
        ):
            _require_exact_numeric_shadow(numeric, exact, role)
    elif isinstance(value, BSplineCurve):
        for numeric, exact, role in (
            (
                value.control_points,
                value.exact_control_point_bank,
                "BSplineCurve.control_points",
            ),
            (value.weights, value.exact_weight_bank, "BSplineCurve.weights"),
            (value.knots, value.exact_knot_bank, "BSplineCurve.knots"),
        ):
            if exact is not None:
                _require_exact_numeric_shadow(numeric, exact, role)
    elif isinstance(value, CylinderPatch):
        for numeric, exact, role in (
            (value.origin, value.origin_exact, "CylinderPatch.origin"),
            (value.first_axis, value.first_axis_exact, "CylinderPatch.first_axis"),
            (value.second_axis, value.second_axis_exact, "CylinderPatch.second_axis"),
            (value.axis, value.axis_exact, "CylinderPatch.axis"),
            (value.radius, value.radius_exact, "CylinderPatch.radius"),
        ):
            _require_exact_numeric_shadow(numeric, exact, role)
    elif isinstance(value, BSplineSurfacePatch):
        for numeric, exact, role in (
            (
                value.control_points,
                value.exact_control_point_bank,
                "BSplineSurfacePatch.control_points",
            ),
            (value.weights, value.exact_weight_bank, "BSplineSurfacePatch.weights"),
            (value.u_knots, value.exact_u_knot_bank, "BSplineSurfacePatch.u_knots"),
            (value.v_knots, value.exact_v_knot_bank, "BSplineSurfacePatch.v_knots"),
        ):
            if exact is not None:
                _require_exact_numeric_shadow(numeric, exact, role)
    elif isinstance(value, TSplineSurfacePatch):
        value.validate_source_bank()
        value.validate_exact_transfer_chain()
    elif isinstance(value, OffsetCurve):
        value.validate_source_bank()
    elif isinstance(value, OffsetSurface):
        _require_exact_numeric_shadow(
            value.distance, value.distance_exact, "OffsetSurface.distance"
        )
        if type(value.frame_sign) is not int or value.frame_sign not in (-1, 0, 1):
            raise ValueError("OffsetSurface.frame_sign is not a valid exact sign.")
        _validate_source_shadows(value.base, visited)
        sphere_witness = (
            _exact_spherical_revolution_witness(value.base)
            if isinstance(value.base, RevolutionSurface)
            else None
        )
        expected_witness = (
            None
            if sphere_witness is None
            else (
                sphere_witness[0],
                sphere_witness[1],
                sphere_witness[2],
                value.frame_sign,
            )
        )
        if value._sphere_revolution_witness != expected_witness:
            raise ValueError(
                "OffsetSurface spherical revolution witness does not match its exact source."
            )
    elif isinstance(value, SurfaceIsoparametricCurve):
        if value.fixed_value_exact is not None:
            _require_exact_numeric_shadow(
                np.asarray(value.fixed_value),
                value.fixed_value_exact,
                "SurfaceIsoparametricCurve.fixed_value",
            )
    elif isinstance(value, ConicRootLinePCurve):
        _validate_conic_parameter_root(value.root, visited)
        _require_exact_numeric_shadow(
            value.height_origin,
            value.height_origin_exact,
            "ConicRootLinePCurve.height_origin",
        )
        _require_exact_numeric_shadow(
            value.height_direction,
            value.height_direction_exact,
            "ConicRootLinePCurve.height_direction",
        )
        lower, upper = _conic_root_phase_enclosure(value.root, value.phase_shift_exact)
        angle = float(np.asarray(value.angle))
        if not lower <= angle <= upper:
            raise ValueError(
                "ConicRootLinePCurve angle shadow does not match its exact phase root."
            )
    elif isinstance(value, PlaneCylinderSectionPCurve):
        for numeric, exact, role in (
            (value.offset, value.offset_exact, "PlaneCylinderSectionPCurve.offset"),
            (value.linear, value.linear_exact, "PlaneCylinderSectionPCurve.linear"),
            (value.cosine, value.cosine_exact, "PlaneCylinderSectionPCurve.cosine"),
            (value.sine, value.sine_exact, "PlaneCylinderSectionPCurve.sine"),
        ):
            _require_exact_numeric_shadow(numeric, exact, role)
        if value.angle_offset_root is not None:
            _validate_conic_parameter_root(value.angle_offset_root, visited)
        expected_phase = (
            0.0 if value.angle_offset_root is None else value.angle_offset_root.parameter
        )
        _require_exact_numeric_shadow(
            value.angle_offset,
            expected_phase,
            "PlaneCylinderSectionPCurve.angle_offset",
        )

    if isinstance(value, AffinePCurve):
        _validate_source_shadows(value.curve, visited)
    elif isinstance(value, PeriodicPCurve):
        _validate_source_shadows(value.source_curve, visited)
    elif isinstance(value, OffsetCurve):
        _validate_source_shadows(value.base, visited)
    elif isinstance(value, OffsetSurface):
        _validate_source_shadows(value.base, visited)
    elif isinstance(value, SurfaceMappedCurve):
        _validate_source_shadows(value.surface, visited)
        _validate_source_shadows(value.pcurve, visited)
    elif isinstance(value, SurfaceIsoparametricCurve):
        _validate_source_shadows(value.surface, visited)
    elif isinstance(value, ConicRootLinePCurve):
        _validate_conic_parameter_root(value.root, visited)
    elif isinstance(value, PlaneCylinderSectionPCurve):
        if value.angle_offset_root is not None:
            _validate_conic_parameter_root(value.angle_offset_root, visited)
    elif isinstance(value, RevolutionSurface):
        _validate_source_shadows(value.curve, visited)
    elif isinstance(value, ExtrusionSurface):
        _validate_source_shadows(value.curve, visited)
    elif isinstance(value, RuledSurface):
        _validate_source_shadows(value.first, visited)
        _validate_source_shadows(value.second, visited)
    elif isinstance(value, CurveTrimSegment):
        _validate_source_shadows(value.curve, visited)
    elif isinstance(value, IntersectionPCurve):
        _validate_source_shadows(value.curve, visited)
    elif isinstance(value, IntersectionCurve):
        _validate_source_shadows(value.first.patch, visited)
        _validate_source_shadows(value.second.patch, visited)
    elif isinstance(value, PlacedCurve):
        _validate_placed_source_shadows(value.rotation, value.translation, value.affine)
        _validate_source_shadows(value.definition, visited)
    elif isinstance(value, PlacedSurface):
        _validate_placed_source_shadows(value.rotation, value.translation, value.affine)
        _validate_source_shadows(value.definition, visited)


def _validate_curve_source_shadows(curve: AbstractCurve, /) -> None:
    if not isinstance(curve, AbstractCurve):
        raise TypeError("curve must be an AbstractCurve.")
    _validate_source_shadows(curve, set())


def _validate_surface_source_shadows(patch: AbstractSurfacePatch, /) -> None:
    if not isinstance(patch, AbstractSurfacePatch):
        raise TypeError("patch must be an AbstractSurfacePatch.")
    _validate_source_shadows(patch, set())


__all__ = [
    "AbstractCurve",
    "AbstractSurfacePatch",
    "BRepCurve",
    "BSplineCurve",
    "BSplineSurfacePatch",
    "bspline_curve_bounds",
    "bspline_curve_derivative_bounds",
    "bspline_surface_derivative_bounds",
    "CircleCurve",
    "ConePatch",
    "CylinderPatch",
    "EllipseCurve",
    "ExtrusionSurface",
    "HyperbolaCurve",
    "ExactLineCurve",
    "LineCurve",
    "OffsetCurve",
    "OffsetSurface",
    "ParabolaCurve",
    "PlaneCylinderSectionPCurve",
    "PlanePatch",
    "RevolutionSurface",
    "RuledSurface",
    "SpherePatch",
    "SurfaceMappedCurve",
    "SurfaceIsoparametricCurve",
    "SurfacePatch",
    "TorusPatch",
    "surface_differential",
    "surface_jacobian",
    "surface_normal",
]
