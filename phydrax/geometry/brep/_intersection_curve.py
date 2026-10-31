#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-defined intersection branches and their coupled chart p-curves."""

from __future__ import annotations

import functools
import math
import weakref
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial
from typing import Any, Callable, cast, Literal, TypeAlias, TypeGuard

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import ExactScalar
from ...typing import checked, parse
from .._interval_enclosure import prepare_interval_function, PreparedIntervalFunction
from ._intersection_kernel import IntervalEvaluator, krawczyk, KrawczykEvaluator
from ._patches import AbstractCurve, AbstractSurfacePatch, CylinderPatch
from ._trim_helpers import period_scalar_bounds, PeriodOffset


IntersectionSide: TypeAlias = Literal["first", "second"]
IntersectionEndpointKind: TypeAlias = Literal[
    "boundary", "singular", "uncertified", "closed"
]

_FREE_AXES = jnp.asarray(((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)), dtype=jnp.int32)


class SurfaceRegion(StrictModule):
    """A source surface restricted to one closed parameter rectangle."""

    patch: AbstractSurfacePatch
    parameter_box: Array

    @checked
    def __init__(self, patch: AbstractSurfacePatch, parameter_box: ArrayLike) -> None:
        box = np.asarray(parameter_box, dtype=np.float64)
        if box.shape != (2, 2):
            raise ValueError("SurfaceRegion.parameter_box must have shape (2, 2).")
        if not np.all(np.isfinite(box)) or np.any(box[1] <= box[0]):
            raise ValueError("SurfaceRegion.parameter_box must be finite and increasing.")
        patch.validate_parameter_box(box)
        self.patch = patch
        self.parameter_box = jnp.asarray(box, dtype=jnp.float64)

    @property
    def lower(self) -> tuple[float, float]:
        lower = np.asarray(self.parameter_box[0])
        return float(lower[0]), float(lower[1])

    @property
    def upper(self) -> tuple[float, float]:
        upper = np.asarray(self.parameter_box[1])
        return float(upper[0]), float(upper[1])

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": "brep-surface-region",
            "patch_type": f"{type(self.patch).__module__}.{type(self.patch).__qualname__}",
            "patch": array_tree_fingerprint(self.patch),
            "parameter_box": array_tree_fingerprint(self.parameter_box),
        }


class SurfacePairSystem(StrictModule):
    """The common spatial zero set of two generating surface maps."""

    first: AbstractSurfacePatch
    second: AbstractSurfacePatch

    def __init__(self, first: AbstractSurfacePatch, second: AbstractSurfacePatch) -> None:
        self.first = first
        self.second = second

    def residual(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        return self.first.evaluate(values[..., :2]) - self.second.evaluate(
            values[..., 2:]
        )


class _NodeSystem(StrictModule):
    pair: SurfacePairSystem
    axis: int = eqx.field(static=True)
    value: float = eqx.field(static=True)

    def __init__(
        self,
        pair: SurfacePairSystem,
        axis: int,
        value: float,
    ) -> None:
        self.pair = pair
        self.axis = axis
        self.value = value

    def residual(self, parameters: Array, /) -> Array:
        return jnp.concatenate(
            (self.pair.residual(parameters), (parameters[self.axis] - self.value)[None])
        )


class _PreparedPairJacobian(StrictModule):
    """Outward pair Jacobian from source derivative enclosures when available."""

    system: SurfacePairSystem
    fallback: PreparedIntervalFunction

    def __init__(self, system: SurfacePairSystem) -> None:
        self.system = system
        self.fallback = prepare_interval_function(jax.jacfwd(system.residual), 4)

    def evaluate(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        if lower_.ndim != 2 or lower_.shape[1] != 4 or upper_.shape != lower_.shape:
            raise ValueError(
                "Surface-pair Jacobian boxes must have aligned shape (count, 4)."
            )
        result_lower = np.empty((lower_.shape[0], 3, 4), dtype=np.float64)
        result_upper = np.empty_like(result_lower)
        try:
            for index in range(lower_.shape[0]):
                first_lower, first_upper = self.system.first.derivative_bounds(
                    np.stack((lower_[index, :2], upper_[index, :2]))
                )
                second_lower, second_upper = self.system.second.derivative_bounds(
                    np.stack((lower_[index, 2:], upper_[index, 2:]))
                )
                first_lower = np.asarray(first_lower, dtype=np.float64)
                first_upper = np.asarray(first_upper, dtype=np.float64)
                second_lower = np.asarray(second_lower, dtype=np.float64)
                second_upper = np.asarray(second_upper, dtype=np.float64)
                if any(
                    values.shape != (3, 2)
                    for values in (first_lower, first_upper, second_lower, second_upper)
                ):
                    raise ValueError("Surface derivative bounds must have shape (3, 2).")
                result_lower[index, :, :2] = first_lower
                result_upper[index, :, :2] = first_upper
                result_lower[index, :, 2:] = -second_upper
                result_upper[index, :, 2:] = -second_lower
        except NotImplementedError:
            return self.fallback.evaluate(lower_, upper_)
        return result_lower, result_upper


class _PreparedPair(StrictModule):
    system: SurfacePairSystem
    value: PreparedIntervalFunction
    jacobian: _PreparedPairJacobian

    def __init__(self, system: SurfacePairSystem) -> None:
        self.system = system
        self.value = prepare_interval_function(system.residual, 4)
        self.jacobian = _PreparedPairJacobian(system)

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        values = jnp.asarray(points, dtype=jnp.float64)
        return np.asarray(
            _batched_curve_jacobians(self.system.first, self.system.second, values),
            dtype=np.float64,
        )


class _RevolutionCylinderPairJacobian:
    """Transport proxy-cylinder derivative bounds to source profile coordinates."""

    def __init__(
        self,
        proxy_jacobian: IntervalEvaluator,
        slopes: tuple[Fraction, Fraction],
        /,
    ) -> None:
        self.proxy_jacobian = proxy_jacobian
        self.slopes = slopes

    def evaluate(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import interval_multiply
        from .._rational import fraction_down, fraction_up

        proxy_lower, proxy_upper = _revolution_cylinder_parameter_boxes(
            lower, upper, self.slopes
        )
        result_lower, result_upper = self.proxy_jacobian.evaluate(
            proxy_lower, proxy_upper
        )
        result_lower = np.asarray(result_lower, dtype=np.float64).copy()
        result_upper = np.asarray(result_upper, dtype=np.float64).copy()
        for axis, slope in ((1, self.slopes[0]), (3, self.slopes[1])):
            slope_interval = (
                np.asarray(fraction_down(slope), dtype=np.float64),
                np.asarray(fraction_up(slope), dtype=np.float64),
            )
            result_lower[:, :, axis], result_upper[:, :, axis] = interval_multiply(
                (result_lower[:, :, axis], result_upper[:, :, axis]),
                slope_interval,
            )
        return result_lower, result_upper


def _revolution_cylinder_parameter_boxes(
    lower: ArrayLike,
    upper: ArrayLike,
    slopes: tuple[Fraction, Fraction],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    from .._interval_enclosure import interval_multiply
    from .._rational import fraction_down, fraction_up

    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        lower_.ndim != 2
        or lower_.shape[1] != 4
        or upper_.shape != lower_.shape
        or np.any(lower_ > upper_)
    ):
        raise ValueError(
            "Revolution-cylinder source boxes must be ordered and shaped (count, 4)."
        )
    proxy_lower, proxy_upper = lower_.copy(), upper_.copy()
    for axis, slope in ((1, slopes[0]), (3, slopes[1])):
        slope_interval = (
            np.asarray(fraction_down(slope), dtype=np.float64),
            np.asarray(fraction_up(slope), dtype=np.float64),
        )
        proxy_lower[:, axis], proxy_upper[:, axis] = interval_multiply(
            (lower_[:, axis], upper_[:, axis]), slope_interval
        )
    return proxy_lower, proxy_upper


class _RevolutionCylinderPairPrepared(StrictModule):
    """Source-bound Krawczyk preparation for exact line-profile revolutions.

    The residual stays in the source JAXPR. Its proxy cylinder uses the exact
    chart ``(u, h) = (u, slope * v)`` only for Jacobian bounds; outward interval
    transport preserves negative slopes. Chart and node inclusions still run
    against the original source regions and source-coordinate boxes.
    """

    system: SurfacePairSystem
    source_regions: tuple[SurfaceRegion, SurfaceRegion]
    source_slopes: tuple[Fraction, Fraction]
    proxy: _PreparedPair
    value: PreparedIntervalFunction
    jacobian: _RevolutionCylinderPairJacobian

    def __init__(
        self,
        system: SurfacePairSystem,
        source_regions: tuple[SurfaceRegion, SurfaceRegion],
        proxy_regions: tuple[SurfaceRegion, SurfaceRegion],
        source_slopes: tuple[Fraction, Fraction],
        /,
    ) -> None:
        if (
            system.first is not source_regions[0].patch
            or system.second is not source_regions[1].patch
        ):
            raise ValueError(
                "Revolution-cylinder preparation must bind the actual source regions."
            )
        proxy_system = SurfacePairSystem(proxy_regions[0].patch, proxy_regions[1].patch)
        proxy = _PreparedPair(proxy_system)
        self.system = system
        self.source_regions = source_regions
        self.source_slopes = source_slopes
        self.proxy = proxy
        self.value = prepare_interval_function(system.residual, 4)
        self.jacobian = _RevolutionCylinderPairJacobian(proxy.jacobian, source_slopes)

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        values = jnp.asarray(points, dtype=jnp.float64)
        return np.asarray(
            _batched_curve_jacobians(self.system.first, self.system.second, values),
            dtype=np.float64,
        )


def _revolution_cylinder_pair_prepared(
    system: SurfacePairSystem,
    source_regions: tuple[SurfaceRegion, SurfaceRegion] | None,
    /,
) -> _RevolutionCylinderPairPrepared | None:
    if source_regions is None or (
        system.first is not source_regions[0].patch
        or system.second is not source_regions[1].patch
    ):
        return None
    from ._intersection import _revolution_cylinder_view

    first, second = (
        _revolution_cylinder_view(source_regions[0]),
        _revolution_cylinder_view(source_regions[1]),
    )
    if first is None or second is None:
        return None
    return _RevolutionCylinderPairPrepared(
        system,
        source_regions,
        (first[0], second[0]),
        (first[1], second[1]),
    )


class _SingularOrthogonalCylinderFamily(ValueError):
    """An exact equal-radius cylinder family has a singular branch crossing."""


class _DisjointOrthogonalCylinderFamily(ValueError):
    """An exact normal separation excludes equal-radius cylinder intersections."""


@dataclass(frozen=True, slots=True)
class _OrthogonalCylinderFamily:
    """Exact two-loop intersection data for two bounded line-revolution cylinders."""

    source_regions: tuple[SurfaceRegion, SurfaceRegion]
    proxy_regions: tuple[SurfaceRegion, SurfaceRegion]
    source_slopes: tuple[Fraction, Fraction]
    large_index: int
    normal: tuple[Fraction, Fraction, Fraction]
    delta_n: Fraction
    delta_large_axis: Fraction
    delta_small_axis: Fraction
    large_radius: Fraction
    small_radius: Fraction
    small_normal_coefficients: tuple[Fraction, Fraction]
    small_large_axis_coefficients: tuple[Fraction, Fraction]

    @property
    def small_index(self) -> int:
        return 1 - self.large_index

    @property
    def large(self) -> CylinderPatch:
        patch = self.proxy_regions[self.large_index].patch
        if not isinstance(patch, CylinderPatch):
            raise TypeError("An orthogonal cylinder family must retain cylinder proxies.")
        return patch

    @property
    def small(self) -> CylinderPatch:
        patch = self.proxy_regions[self.small_index].patch
        if not isinstance(patch, CylinderPatch):
            raise TypeError("An orthogonal cylinder family must retain cylinder proxies.")
        return patch


type _FamilyRecord = (
    _OrthogonalCylinderFamily
    | _DisjointOrthogonalCylinderFamily
    | _SingularOrthogonalCylinderFamily
    | None
)

# Immutable source regions keyed by object identity. A stored family holds its
# derived proxy regions in place of the keyed regions, so no record keeps its
# own key alive; each entry is dropped when either keyed region is collected.
_FAMILY_RECORDS: dict[tuple[int, int], _FamilyRecord] = {}


def orthogonal_cylinder_family(
    source_a: SurfaceRegion,
    source_b: SurfaceRegion,
    /,
) -> _OrthogonalCylinderFamily | None:
    """Certify regular unequal branches; classify equal-radius singularity or separation.

    The exact derivation is pure in the two immutable source regions, so it is
    prepared once per region pair and reused by every branch query.
    """
    key = id(source_a), id(source_b)
    record = _FAMILY_RECORDS.get(key)
    if key not in _FAMILY_RECORDS:
        try:
            family = _derive_orthogonal_cylinder_family(source_a, source_b)
        except (
            _DisjointOrthogonalCylinderFamily,
            _SingularOrthogonalCylinderFamily,
        ) as error:
            # A fresh refusal carries no traceback frames that reference the keys.
            record = type(error)(*error.args)
        else:
            record = (
                None
                if family is None
                else replace(family, source_regions=family.proxy_regions)
            )
        if _region_is_concrete(source_a) and _region_is_concrete(source_b):
            _FAMILY_RECORDS[key] = record
            weakref.finalize(source_a, _FAMILY_RECORDS.pop, key, None)
            weakref.finalize(source_b, _FAMILY_RECORDS.pop, key, None)
    if record is None:
        return None
    if isinstance(record, _DisjointOrthogonalCylinderFamily):
        raise _DisjointOrthogonalCylinderFamily(*record.args)
    if isinstance(record, _SingularOrthogonalCylinderFamily):
        raise _SingularOrthogonalCylinderFamily(*record.args)
    return replace(record, source_regions=(source_a, source_b))


def _region_is_concrete(region: SurfaceRegion, /) -> bool:
    return not any(isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(region))


def _derive_orthogonal_cylinder_family(
    source_a: SurfaceRegion,
    source_b: SurfaceRegion,
    /,
) -> _OrthogonalCylinderFamily | None:
    """Derive regular unequal branches; classify equal-radius singularity or separation."""
    from .._rational import rational_cross, rational_dot
    from ._intersection import _revolution_cylinder_view
    from ._patches import CylinderPatch, ExactLineCurve, LineCurve, RevolutionSurface

    regions = source_a, source_b
    for region in regions:
        patch = region.patch
        if not isinstance(patch, RevolutionSurface) or not isinstance(
            patch.curve, (LineCurve, ExactLineCurve)
        ):
            return None
        profile_arrays = (
            (patch.curve.origin, patch.curve.direction)
            if isinstance(patch.curve, LineCurve)
            else (patch.curve.first, patch.curve.last)
        )
        if any(
            isinstance(value, Tracer)
            for value in (
                region.parameter_box,
                patch.axis_origin,
                patch.axis_direction,
                *profile_arrays,
            )
        ):
            return None

    first_view = _revolution_cylinder_view(source_a)
    second_view = _revolution_cylinder_view(source_b)
    if first_view is None or second_view is None:
        return None
    proxy_regions = first_view[0], second_view[0]
    source_slopes = first_view[1], second_view[1]
    first_patch, second_patch = proxy_regions[0].patch, proxy_regions[1].patch
    if not isinstance(first_patch, CylinderPatch) or not isinstance(
        second_patch, CylinderPatch
    ):
        return None
    first, second = first_patch, second_patch
    patches = first, second

    def fraction_vector(
        values: tuple[ExactScalar, ...],
    ) -> tuple[Fraction, ...] | None:
        if not all(isinstance(value, Fraction) for value in values):
            return None
        return cast(tuple[Fraction, ...], tuple(values))

    vectors = (
        fraction_vector(first.origin_exact),
        fraction_vector(first.axis_exact),
        fraction_vector(first.first_axis_exact),
        fraction_vector(first.second_axis_exact),
        fraction_vector(second.origin_exact),
        fraction_vector(second.axis_exact),
        fraction_vector(second.first_axis_exact),
        fraction_vector(second.second_axis_exact),
    )
    radii = fraction_vector((first.radius_exact, second.radius_exact))
    if radii is None or any(vector is None for vector in vectors):
        return None
    (
        first_origin,
        first_axis,
        first_radial,
        first_tangent,
        second_origin,
        second_axis,
        second_radial,
        second_tangent,
    ) = cast(
        tuple[
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
        ],
        vectors,
    )
    if (
        rational_dot(first_axis, first_axis) != 1
        or rational_dot(second_axis, second_axis) != 1
        or rational_dot(first_axis, second_axis) != 0
        or rational_dot(first_radial, first_radial) != 1
        or rational_dot(first_tangent, first_tangent) != 1
        or rational_dot(second_radial, second_radial) != 1
        or rational_dot(second_tangent, second_tangent) != 1
        or rational_dot(first_radial, first_tangent) != 0
        or rational_dot(second_radial, second_tangent) != 0
        or rational_cross(first_radial, first_tangent) != first_axis
        or rational_cross(second_radial, second_tangent) != second_axis
    ):
        return None
    normal = rational_cross(first_axis, second_axis)
    if rational_dot(normal, normal) != 1:
        return None
    if min(radii) <= 0:
        return None
    equal_radii = radii[0] == radii[1]
    large_index = 0 if equal_radii or radii[0] > radii[1] else 1
    small_index = 1 - large_index
    large_patch, small_patch = patches[large_index], patches[small_index]
    if not isinstance(large_patch, CylinderPatch) or not isinstance(
        small_patch, CylinderPatch
    ):
        return None
    large_axis = fraction_vector(large_patch.axis_exact)
    small_axis = fraction_vector(small_patch.axis_exact)
    large_origin = fraction_vector(large_patch.origin_exact)
    small_origin = fraction_vector(small_patch.origin_exact)
    small_radial = fraction_vector(small_patch.first_axis_exact)
    small_tangent = fraction_vector(small_patch.second_axis_exact)
    if any(
        value is None
        for value in (
            large_axis,
            small_axis,
            large_origin,
            small_origin,
            small_radial,
            small_tangent,
        )
    ):
        return None
    large_axis = cast(tuple[Fraction, ...], large_axis)
    small_axis = cast(tuple[Fraction, ...], small_axis)
    large_origin = cast(tuple[Fraction, ...], large_origin)
    small_origin = cast(tuple[Fraction, ...], small_origin)
    small_radial = cast(tuple[Fraction, ...], small_radial)
    small_tangent = cast(tuple[Fraction, ...], small_tangent)
    normal = rational_cross(large_axis, small_axis)
    if rational_dot(large_axis, small_axis) != 0 or rational_dot(normal, normal) != 1:
        return None
    delta = tuple(
        end - start for start, end in zip(large_origin, small_origin, strict=True)
    )
    delta_n = rational_dot(delta, normal)
    delta_large_axis = rational_dot(delta, large_axis)
    delta_small_axis = rational_dot(delta, small_axis)
    large_radius = radii[large_index]
    small_radius = radii[small_index]
    if not equal_radii and abs(delta_n) + small_radius >= large_radius:
        return None
    large_region = regions[large_index]
    small_region = regions[small_index]
    for region in (large_region, small_region):
        patch = region.patch
        if (
            patch.native_period_symbols()[0] != "two_pi"
            or patch.periods[0] is None
            or Fraction.from_float(float(region.parameter_box[1, 0]))
            - Fraction.from_float(float(region.parameter_box[0, 0]))
            != Fraction.from_float(float(patch.periods[0]))
        ):
            return None
    large_slope, small_slope = (
        source_slopes[large_index],
        source_slopes[small_index],
    )
    if large_slope == 0 or small_slope == 0:
        return None
    large_profile = large_region.parameter_box
    small_profile = small_region.parameter_box
    large_height = (
        delta_large_axis - small_radius,
        delta_large_axis + small_radius,
    )
    large_parameter = tuple(value / large_slope for value in large_height)
    if min(large_parameter) < Fraction.from_float(float(large_profile[0, 1])) or max(
        large_parameter
    ) > Fraction.from_float(float(large_profile[1, 1])):
        return None
    small_height = (
        -large_radius - delta_small_axis,
        large_radius - delta_small_axis,
    )
    small_parameter = tuple(value / small_slope for value in small_height)
    if min(small_parameter) < Fraction.from_float(float(small_profile[0, 1])) or max(
        small_parameter
    ) > Fraction.from_float(float(small_profile[1, 1])):
        return None
    small_normal_coefficients = (
        rational_dot(small_radial, normal),
        rational_dot(small_tangent, normal),
    )
    small_large_axis_coefficients = (
        rational_dot(small_radial, large_axis),
        rational_dot(small_tangent, large_axis),
    )
    if (
        sum(value * value for value in small_normal_coefficients) != 1
        or sum(value * value for value in small_large_axis_coefficients) != 1
    ):
        return None
    if equal_radii:
        if abs(delta_n) <= 2 * large_radius:
            raise _SingularOrthogonalCylinderFamily(
                "Equal-radius orthogonal cylinders have a singular intersection branch."
            )
        raise _DisjointOrthogonalCylinderFamily(
            "Equal-radius orthogonal cylinders are separated in the normal direction."
        )
    return _OrthogonalCylinderFamily(
        regions,
        proxy_regions,
        source_slopes,
        large_index,
        normal,
        delta_n,
        delta_large_axis,
        delta_small_axis,
        large_radius,
        small_radius,
        small_normal_coefficients,
        small_large_axis_coefficients,
    )


@functools.cache
def interval_context(decimal_digits: int, /) -> Any:
    """Return one shared outward mpmath interval context at a fixed precision.

    Context construction wraps every special function and dominates small exact
    enclosures; a context is never mutated after this fixed-precision setup.
    """
    from mpmath.ctx_iv import MPIntervalContext

    context = MPIntervalContext()
    context.dps = decimal_digits
    return context


def orthogonal_cylinder_graph_bounds(
    family: _OrthogonalCylinderFamily,
    theta_lower: float,
    theta_upper: float,
    branch_sign: int,
    first_angle_reference: float,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Outward-enclose the closed-form source p-curve over a source-angle interval."""

    if (
        branch_sign not in (-1, 1)
        or not math.isfinite(theta_lower)
        or not math.isfinite(theta_upper)
        or theta_lower > theta_upper
        or not math.isfinite(first_angle_reference)
    ):
        return None
    large, small = family.large, family.small
    if not isinstance(large, CylinderPatch) or not isinstance(small, CylinderPatch):
        return None
    large_first = cast(tuple[Fraction, ...], large.first_axis_exact)
    large_second = cast(tuple[Fraction, ...], large.second_axis_exact)
    small_axis = cast(tuple[Fraction, ...], small.axis_exact)
    cosine_angle = (
        sum(
            (a * b for a, b in zip(small_axis, large_first, strict=True)),
            Fraction(0),
        ),
        sum(
            (a * b for a, b in zip(family.normal, large_first, strict=True)),
            Fraction(0),
        ),
    )
    sine_angle = (
        sum(
            (a * b for a, b in zip(small_axis, large_second, strict=True)),
            Fraction(0),
        ),
        sum(
            (a * b for a, b in zip(family.normal, large_second, strict=True)),
            Fraction(0),
        ),
    )
    context = interval_context(80)

    def scalar(value: Fraction) -> Any:
        return context.mpf(value.numerator) / context.mpf(value.denominator)

    def interval(first: Fraction, last: Fraction) -> Any:
        return context.mpf((scalar(first), scalar(last)))

    first_parameter = Fraction.from_float(float(theta_lower))
    last_parameter = Fraction.from_float(float(theta_upper))
    theta = interval(first_parameter, last_parameter)
    cosine, sine = context.cos(theta), context.sin(theta)
    radius_large = scalar(family.large_radius)
    radius_small = scalar(family.small_radius)
    delta_n = scalar(family.delta_n)
    delta_a = scalar(family.delta_large_axis)
    delta_b = scalar(family.delta_small_axis)
    normal_cosine, normal_sine = (
        scalar(value) for value in family.small_normal_coefficients
    )
    axial_cosine, axial_sine = (
        scalar(value) for value in family.small_large_axis_coefficients
    )
    radial_n = delta_n + radius_small * (normal_cosine * cosine + normal_sine * sine)
    axial = delta_a + radius_small * (axial_cosine * cosine + axial_sine * sine)
    radicand = radius_large**2 - radial_n**2
    if float(radicand.a) <= 0.0 or not math.isfinite(float(radicand.b)):
        return None
    radial_large = context.sqrt(radicand)
    if branch_sign < 0:
        radial_large = -radial_large
    small_height = radial_large - delta_b

    midpoint_parameter = (first_parameter + last_parameter) / 2
    theta_midpoint = scalar(midpoint_parameter)
    midpoint_cosine = context.cos(theta_midpoint)
    midpoint_sine = context.sin(theta_midpoint)
    midpoint_n = delta_n + radius_small * (
        normal_cosine * midpoint_cosine + normal_sine * midpoint_sine
    )
    midpoint_radicand = radius_large**2 - midpoint_n**2
    if float(midpoint_radicand.a) <= 0.0:
        return None
    midpoint_x = context.sqrt(midpoint_radicand)
    if branch_sign < 0:
        midpoint_x = -midpoint_x
    cosine_phi = (
        scalar(cosine_angle[0]) * midpoint_x + scalar(cosine_angle[1]) * midpoint_n
    ) / radius_large
    sine_phi = (
        scalar(sine_angle[0]) * midpoint_x + scalar(sine_angle[1]) * midpoint_n
    ) / radius_large
    phi_midpoint = context.atan2(sine_phi, cosine_phi)
    phi_numeric = 0.5 * (float(phi_midpoint.a) + float(phi_midpoint.b))
    turns = round((first_angle_reference - phi_numeric) / math.tau)
    phi_midpoint += turns * 2 * context.pi

    radial_n_derivative = radius_small * (-normal_cosine * sine + normal_sine * cosine)
    phi_derivative = radial_n_derivative / radial_large
    phi = phi_midpoint + phi_derivative * (theta - theta_midpoint)
    large_slope = scalar(family.source_slopes[family.large_index])
    small_slope = scalar(family.source_slopes[family.small_index])
    large_profile = axial / large_slope
    small_profile = small_height / small_slope
    source_intervals = (
        (phi, large_profile, theta, small_profile)
        if family.large_index == 0
        else (theta, small_profile, phi, large_profile)
    )

    def float_bounds(value: Any) -> tuple[float, float] | None:
        lower, upper = float(value.a), float(value.b)
        if not math.isfinite(lower) or not math.isfinite(upper):
            return None
        return (
            float(np.nextafter(lower, -np.inf)),
            float(np.nextafter(upper, np.inf)),
        )

    bounds = tuple(float_bounds(value) for value in source_intervals)
    if any(value is None for value in bounds):
        return None
    ordered = cast(tuple[tuple[float, float], ...], bounds)
    lower = np.asarray(tuple(value[0] for value in ordered), dtype=np.float64)
    upper = np.asarray(tuple(value[1] for value in ordered), dtype=np.float64)
    if np.any(lower >= upper):
        return None
    return lower, upper


def orthogonal_cylinder_closing_root_bounds(
    family: _OrthogonalCylinderFamily,
    start_value: float,
    branch_sign: int,
    first_angle_reference: float,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Outward-enclose the source root at the exact one-turn image of a start node.

    A closed graph branch ends on the float image of its start angle plus the
    small cylinder's native two-pi period. The exact deck image differs from
    that float by rounding, and wherever the large-cylinder angle moves with the
    small angle (the loop crosses the large chart's seam there), the slice root
    and the exact image root differ in every transverse coordinate. The
    closing node must enclose the exact image so its period turn is proved.
    """
    image_lower, image_upper = period_scalar_bounds(
        PeriodOffset(Fraction.from_float(float(start_value)), Fraction(1))
    )
    return orthogonal_cylinder_graph_bounds(
        family,
        float(image_lower),
        float(image_upper),
        branch_sign,
        first_angle_reference,
    )


@dataclass(frozen=True, slots=True)
class OrthogonalCylinderSeamCrossing:
    """One crossing of a branch through the large cylinder's authored seam.

    ``theta_lower``/``theta_upper`` enclose the small angle of the crossing in
    the small chart window. A crossing on the small chart's own seam is the
    branch's start node, proved to lie exactly on both seams.
    """

    theta_lower: float
    theta_upper: float
    start: bool


def _large_seam_frame(
    family: _OrthogonalCylinderFamily, /
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction], Fraction]:
    """Return the small-axis and normal coordinates of the large frame and seam."""
    from .._rational import rational_dot

    large_first = cast(tuple[Fraction, ...], family.large.first_axis_exact)
    large_second = cast(tuple[Fraction, ...], family.large.second_axis_exact)
    small_axis = cast(tuple[Fraction, ...], family.small.axis_exact)
    cosine = (
        rational_dot(small_axis, large_first),
        rational_dot(family.normal, large_first),
    )
    sine = (
        rational_dot(small_axis, large_second),
        rational_dot(family.normal, large_second),
    )
    seam = Fraction.from_float(
        float(family.source_regions[family.large_index].parameter_box[0, 0])
    )
    return cosine, sine, seam


def orthogonal_cylinder_exact_seam_start(
    family: _OrthogonalCylinderFamily, branch_sign: int, /
) -> bool:
    """Prove exactly that a branch's start root lies on the large cylinder's seam.

    The start root sits on the small chart's seam. Both seams are rational only
    at angle zero, where the large radial direction of the root has rational
    small-axis and normal coordinates ``(x, n)`` with ``x`` the branch-signed
    square root; the proof is then an exact rational identity.
    """
    if branch_sign not in (-1, 1):
        return False
    small_region = family.source_regions[family.small_index]
    cosine, sine, seam = _large_seam_frame(family)
    if seam != 0 or Fraction.from_float(float(small_region.parameter_box[0, 0])) != 0:
        return False
    # At seam angle zero the seam direction is the large first axis.
    normal_x, normal_n = sine
    along_x, along_n = cosine
    if normal_n == 0:
        return False
    radius = family.large_radius
    x = branch_sign * radius * abs(normal_n)
    n = -branch_sign * (1 if normal_n > 0 else -1) * normal_x * radius
    start_n = family.delta_n + family.small_radius * family.small_normal_coefficients[0]
    return start_n == n and along_x * x + along_n * n > 0


def orthogonal_cylinder_seam_crossings(
    family: _OrthogonalCylinderFamily, branch_sign: int, /
) -> tuple[OrthogonalCylinderSeamCrossing, ...] | None:
    """Enclose every crossing of one branch through the large cylinder's seam.

    On a branch the large radial direction has coordinates ``(x, n)`` in the
    (small axis, normal) frame, ``x`` the branch-signed root of ``R^2 - n^2``.
    It points along the seam only where its seam-normal component vanishes
    with a positive seam component, which fixes ``n``; the normal offset
    ``n(theta) = delta_n + r cos(theta - phase)`` then meets that value at most
    twice. Returns ``None`` when an interval sign stays undecided.
    """
    if branch_sign not in (-1, 1):
        return None
    context = interval_context(80)

    def scalar(value: Fraction) -> Any:
        return context.mpf(value.numerator) / context.mpf(value.denominator)

    cosine, sine, seam = _large_seam_frame(family)
    seam_cosine, seam_sine = context.cos(scalar(seam)), context.sin(scalar(seam))
    normal_x = -seam_sine * scalar(cosine[0]) + seam_cosine * scalar(sine[0])
    normal_n = -seam_sine * scalar(cosine[1]) + seam_cosine * scalar(sine[1])
    along_x = seam_cosine * scalar(cosine[0]) + seam_sine * scalar(sine[0])
    along_n = seam_cosine * scalar(cosine[1]) + seam_sine * scalar(sine[1])
    if normal_n.a == 0 and normal_n.b == 0:
        return ()
    if normal_n.a <= 0 <= normal_n.b:
        return None
    normal_sign = 1 if normal_n.a > 0 else -1
    radius = scalar(family.large_radius)
    x = branch_sign * normal_sign * normal_n * radius
    n = -branch_sign * normal_sign * normal_x * radius
    along = along_x * x + along_n * n
    if along.b < 0:
        return ()
    if along.a <= 0:
        return None
    offset_cosine = (n - scalar(family.delta_n)) / scalar(family.small_radius)
    if offset_cosine.b < -1 or offset_cosine.a > 1:
        return ()
    if not (-1 < offset_cosine.a and offset_cosine.b < 1):
        return None
    normal_cosine, normal_sine = (
        scalar(value) for value in family.small_normal_coefficients
    )
    phase = context.atan2(normal_sine, normal_cosine)
    offset = context.atan2(context.sqrt(1 - offset_cosine**2), offset_cosine)
    small_region = family.source_regions[family.small_index]
    window = Fraction.from_float(float(small_region.parameter_box[0, 0]))
    window_lower = scalar(window)
    window_upper = window_lower + 2 * context.pi
    crossings = []
    for theta in (phase - offset, phase + offset):
        middle = 0.5 * (float(theta.a) + float(theta.b))
        theta = theta + math.ceil((float(window) - middle) / math.tau) * 2 * context.pi
        if theta.b < window_lower:
            theta = theta + 2 * context.pi
        if theta.a > window_upper:
            theta = theta - 2 * context.pi
        touches_start = (theta.a <= window_lower <= theta.b) or (
            theta.a <= window_upper <= theta.b
        )
        if touches_start:
            if not orthogonal_cylinder_exact_seam_start(family, branch_sign):
                return None
            crossings.append(
                OrthogonalCylinderSeamCrossing(float(window), float(window), True)
            )
            continue
        if not (window_lower < theta.a and theta.b < window_upper):
            return None
        lower = float(np.nextafter(float(theta.a), -np.inf))
        upper = float(np.nextafter(float(theta.b), np.inf))
        crossings.append(OrthogonalCylinderSeamCrossing(lower, upper, False))
    return tuple(sorted(crossings, key=lambda item: item.theta_lower))


def orthogonal_cylinder_seam_node_certified(
    family: _OrthogonalCylinderFamily,
    branch_sign: int,
    seam_value: float,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> bool:
    """Prove a box isolates the one branch root on a fixed large-angle slice.

    The box's small-angle range is the free variable: the graph over it lies in
    the box, the large angle is strictly monotone there, and its two end slices
    bracket the fixed value strictly, so exactly one branch root has it.
    """
    theta_axis = 2 * family.small_index
    angle_axis = 2 * family.large_index
    theta_lower, theta_upper = float(lower[theta_axis]), float(upper[theta_axis])
    if not theta_lower < theta_upper:
        return False
    reference = 0.5 * float(lower[angle_axis]) + 0.5 * float(upper[angle_axis])
    graph = orthogonal_cylinder_graph_bounds(
        family, theta_lower, theta_upper, branch_sign, reference
    )
    derivative = orthogonal_cylinder_graph_derivative_bounds(
        family, theta_lower, theta_upper, branch_sign
    )
    first = orthogonal_cylinder_graph_bounds(
        family, theta_lower, theta_lower, branch_sign, reference
    )
    last = orthogonal_cylinder_graph_bounds(
        family, theta_upper, theta_upper, branch_sign, reference
    )
    if graph is None or derivative is None or first is None or last is None:
        return False
    free = np.asarray([axis for axis in range(4) if axis != theta_axis], dtype=np.int64)
    if np.any(graph[0][free] < lower[free]) or np.any(graph[1][free] > upper[free]):
        return False
    value = Fraction.from_float(float(seam_value))
    if derivative[0][angle_axis] > 0:
        return (
            Fraction.from_float(float(first[1][angle_axis])) < value
            and Fraction.from_float(float(last[0][angle_axis])) > value
        )
    if derivative[1][angle_axis] < 0:
        return (
            Fraction.from_float(float(first[0][angle_axis])) > value
            and Fraction.from_float(float(last[1][angle_axis])) < value
        )
    return False


def orthogonal_cylinder_graph_derivative_bounds(
    family: _OrthogonalCylinderFamily,
    theta_lower: float,
    theta_upper: float,
    branch_sign: int,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Outward-enclose source p-curve derivatives with respect to small angle."""
    if (
        branch_sign not in (-1, 1)
        or not math.isfinite(theta_lower)
        or not math.isfinite(theta_upper)
        or theta_lower > theta_upper
    ):
        return None
    context = interval_context(80)

    def scalar(value: Fraction) -> Any:
        return context.mpf(value.numerator) / context.mpf(value.denominator)

    def interval(first: Fraction, last: Fraction) -> Any:
        return context.mpf((scalar(first), scalar(last)))

    first_parameter = Fraction.from_float(float(theta_lower))
    last_parameter = Fraction.from_float(float(theta_upper))
    theta = interval(first_parameter, last_parameter)
    cosine, sine = context.cos(theta), context.sin(theta)
    radius_large = scalar(family.large_radius)
    radius_small = scalar(family.small_radius)
    delta_n = scalar(family.delta_n)
    normal_cosine, normal_sine = (
        scalar(value) for value in family.small_normal_coefficients
    )
    axial_cosine, axial_sine = (
        scalar(value) for value in family.small_large_axis_coefficients
    )
    radial_n = delta_n + radius_small * (normal_cosine * cosine + normal_sine * sine)
    radial_n_derivative = radius_small * (-normal_cosine * sine + normal_sine * cosine)
    radicand = radius_large**2 - radial_n**2
    if float(radicand.a) <= 0.0 or not math.isfinite(float(radicand.b)):
        return None
    radial_large = context.sqrt(radicand)
    if branch_sign < 0:
        radial_large = -radial_large
    axial_derivative = radius_small * (-axial_cosine * sine + axial_sine * cosine)
    large_slope = scalar(family.source_slopes[family.large_index])
    small_slope = scalar(family.source_slopes[family.small_index])
    derivatives = (
        radial_n_derivative / radial_large,
        axial_derivative / large_slope,
        context.mpf(1),
        (-radial_n * radial_n_derivative / radial_large) / small_slope,
    )
    source_derivatives = (
        derivatives
        if family.large_index == 0
        else (
            derivatives[2],
            derivatives[3],
            derivatives[0],
            derivatives[1],
        )
    )

    def float_bounds(value: Any) -> tuple[float, float] | None:
        lower, upper = float(value.a), float(value.b)
        if not math.isfinite(lower) or not math.isfinite(upper):
            return None
        return (
            float(np.nextafter(lower, -np.inf)),
            float(np.nextafter(upper, np.inf)),
        )

    bounds = tuple(float_bounds(value) for value in source_derivatives)
    if any(value is None for value in bounds):
        return None
    ordered = cast(tuple[tuple[float, float], ...], bounds)
    lower = np.asarray(tuple(value[0] for value in ordered), dtype=np.float64)
    upper = np.asarray(tuple(value[1] for value in ordered), dtype=np.float64)
    return lower, upper


def orthogonal_cylinder_graph_point(
    family: _OrthogonalCylinderFamily,
    theta: float,
    branch_sign: int,
    first_angle_reference: float,
    /,
) -> np.ndarray | None:
    """Return a numeric seed inside the outward analytic source-root enclosure."""
    bounds = orthogonal_cylinder_graph_bounds(
        family, theta, theta, branch_sign, first_angle_reference
    )
    if bounds is None:
        return None
    lower, upper = bounds
    point = lower + 0.5 * (upper - lower)
    point[2 * family.small_index] = float(theta)
    return point


def _orthogonal_cylinder_branch_sign(
    family: _OrthogonalCylinderFamily,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> int | None:
    """Infer a branch sign only when the source-root box excludes the axis plane."""
    profile_axis = 2 * family.small_index + 1
    slope = family.source_slopes[family.small_index]
    signs = []
    for lower_point, upper_point in zip(lower, upper, strict=True):
        parameter_bounds = (
            Fraction.from_float(float(lower_point[profile_axis])) * slope,
            Fraction.from_float(float(upper_point[profile_axis])) * slope,
        )
        axial_bounds = tuple(
            family.delta_small_axis + value for value in parameter_bounds
        )
        axial_lower, axial_upper = min(axial_bounds), max(axial_bounds)
        if axial_lower > 0:
            signs.append(1)
        elif axial_upper < 0:
            signs.append(-1)
        else:
            return None
    if not signs or any(value != signs[0] for value in signs[1:]):
        return None
    return signs[0]


def _orthogonal_cylinder_atlas_layout(
    family: _OrthogonalCylinderFamily,
    chart_axes: np.ndarray,
    chart_start: np.ndarray,
    chart_end: np.ndarray,
    node_axes: np.ndarray,
    node_values: np.ndarray,
    node_lower: np.ndarray,
    node_upper: np.ndarray,
    /,
) -> bool:
    """Check the chart and node layout of a closed closed-form graph atlas.

    Charts advance the small angle. Nodes fix the small angle, except crossings
    of the large cylinder's seam, which fix the large angle at the exact seam
    value. The start and closing nodes fix the small angle one native period
    apart.
    """
    theta_axis = 2 * family.small_index
    angle_axis = 2 * family.large_index
    small_region = family.source_regions[family.small_index]
    large_region = family.source_regions[family.large_index]
    small_period = small_region.patch.periods[0]
    charts = len(chart_axes)
    if (
        small_period is None
        or charts < 1
        or node_values.shape != (charts + 1,)
        or not np.all(chart_axes == theta_axis)
        or int(node_axes[0]) != theta_axis
        or int(node_axes[-1]) != theta_axis
        or Fraction.from_float(float(node_values[0]))
        != Fraction.from_float(float(small_region.parameter_box[0, 0]))
        or Fraction.from_float(float(node_values[-1]))
        - Fraction.from_float(float(node_values[0]))
        != Fraction.from_float(float(small_period))
    ):
        return False
    seam = Fraction.from_float(float(large_region.parameter_box[0, 0]))
    for node in range(charts + 1):
        thetas = [
            *((float(chart_end[node - 1, theta_axis]),) if node > 0 else ()),
            *((float(chart_start[node, theta_axis]),) if node < charts else ()),
        ]
        if int(node_axes[node]) == theta_axis:
            if any(value != float(node_values[node]) for value in thetas):
                return False
        elif int(node_axes[node]) == angle_axis:
            if (
                Fraction.from_float(float(node_values[node])) != seam
                or len(set(thetas)) != 1
                or not node_lower[node, theta_axis]
                <= thetas[0]
                <= node_upper[node, theta_axis]
            ):
                return False
        else:
            return False
    return all(
        chart_start[chart, theta_axis] < chart_end[chart, theta_axis]
        for chart in range(charts)
    )


def _orthogonal_cylinder_curve_branch_sign(
    curve: IntersectionCurve,
    family: _OrthogonalCylinderFamily,
    /,
) -> int | None:
    """Revalidate the exact source-chart atlas before using its closed form."""
    if (
        not curve.closed
        or not curve.fully_certified
        or curve.start_kind != "closed"
        or curve.end_kind != "closed"
    ):
        return None
    node_lower = np.asarray(curve.node_lower, dtype=np.float64)
    node_upper = np.asarray(curve.node_upper, dtype=np.float64)
    if not _orthogonal_cylinder_atlas_layout(
        family,
        np.asarray(curve.chart_axes),
        np.asarray(curve.chart_start, dtype=np.float64),
        np.asarray(curve.chart_end, dtype=np.float64),
        np.asarray(curve.node_axes),
        np.asarray(curve.node_values, dtype=np.float64),
        node_lower,
        node_upper,
    ):
        return None
    return _orthogonal_cylinder_branch_sign(family, node_lower, node_upper)


def _orthogonal_cylinder_chart_angle_data(
    curve: IntersectionCurve,
    family: _OrthogonalCylinderFamily,
    chart: int,
    branch_sign: int,
    /,
) -> tuple[int, int, int] | None:
    """Prepare a continuous arctangent chart with exact frame sign evidence."""
    from .._rational import rational_dot

    large = family.large
    small = family.small
    large_first = cast(tuple[Fraction, ...], large.first_axis_exact)
    large_second = cast(tuple[Fraction, ...], large.second_axis_exact)
    small_axis = cast(tuple[Fraction, ...], small.axis_exact)
    cosine_coefficients = (
        rational_dot(small_axis, large_first),
        rational_dot(family.normal, large_first),
    )
    sine_coefficients = (
        rational_dot(small_axis, large_second),
        rational_dot(family.normal, large_second),
    )
    if sine_coefficients[1] == 0 and abs(sine_coefficients[0]) == 1:
        mode = 1
        angle_sign = branch_sign * (1 if sine_coefficients[0] > 0 else -1)
    elif cosine_coefficients[1] == 0 and abs(cosine_coefficients[0]) == 1:
        mode = 0
        angle_sign = branch_sign * (1 if cosine_coefficients[0] > 0 else -1)
    else:
        return None
    theta_axis = 2 * family.small_index
    angle_axis = 2 * family.large_index
    start = np.asarray(curve.chart_start[chart], dtype=np.float64)
    end = np.asarray(curve.chart_end[chart], dtype=np.float64)
    theta = 0.5 * float(start[theta_axis]) + 0.5 * float(end[theta_axis])
    point = orthogonal_cylinder_graph_point(
        family, theta, branch_sign, float(start[angle_axis])
    )
    if point is None:
        return None
    cosine, sine = math.cos(theta), math.sin(theta)
    normal_cosine, normal_sine = family.small_normal_coefficients
    radial_n = float(family.delta_n) + float(family.small_radius) * (
        float(normal_cosine) * cosine + float(normal_sine) * sine
    )
    radicand = float(family.large_radius**2) - radial_n * radial_n
    if not math.isfinite(radicand) or radicand <= 0.0:
        return None
    radial_large = branch_sign * math.sqrt(radicand)
    first_component = (
        float(cosine_coefficients[0]) * radial_large
        + float(cosine_coefficients[1]) * radial_n
    )
    second_component = (
        float(sine_coefficients[0]) * radial_large
        + float(sine_coefficients[1]) * radial_n
    )
    if mode == 1:
        if second_component == 0.0:
            return None
        angle = angle_sign * 0.5 * math.pi - math.atan(first_component / second_component)
    else:
        if first_component == 0.0:
            return None
        angle = math.atan(second_component / first_component)
        if angle_sign < 0:
            angle += math.pi
    turns = int(round((float(point[angle_axis]) - angle) / math.tau))
    return mode, angle_sign, turns


def orthogonal_cylinder_branches_disjoint(
    first: IntersectionCurve,
    second: IntersectionCurve,
    /,
) -> bool:
    """Prove opposite certified branches of one cylinder pair cannot meet."""
    if (
        first is second
        or not first.closed
        or not second.closed
        or not first.fully_certified
        or not second.fully_certified
    ):
        return False
    first_family = orthogonal_cylinder_family(first.first, first.second)
    second_family = orthogonal_cylinder_family(second.first, second.second)
    if (
        first_family is None
        or second_family is None
        or any(
            left is not right
            for left, right in zip(
                first_family.source_regions,
                second_family.source_regions,
                strict=True,
            )
        )
    ):
        return False
    first_sign = _orthogonal_cylinder_curve_branch_sign(first, first_family)
    second_sign = _orthogonal_cylinder_curve_branch_sign(second, second_family)
    return (
        first_sign is not None and second_sign is not None and first_sign != second_sign
    )


def _orthogonal_cylinder_curve_certificate(
    family: _OrthogonalCylinderFamily,
    prepared: KrawczykEvaluator,
    chart_axes: np.ndarray,
    chart_start: np.ndarray,
    chart_end: np.ndarray,
    chart_lower: np.ndarray,
    chart_upper: np.ndarray,
    node_axes: np.ndarray,
    node_values: np.ndarray,
    node_lower: np.ndarray,
    node_upper: np.ndarray,
    closed: bool,
    start_kind: IntersectionEndpointKind,
    end_kind: IntersectionEndpointKind,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Verify the exact closed-form graph atlas and prepare branch evaluation data."""
    if (
        not isinstance(prepared, _RevolutionCylinderPairPrepared)
        or prepared.source_regions[0] is not family.source_regions[0]
        or prepared.source_regions[1] is not family.source_regions[1]
        or not closed
        or start_kind != "closed"
        or end_kind != "closed"
    ):
        return None
    theta_axis = 2 * family.small_index
    large_angle_axis = 2 * family.large_index
    if not _orthogonal_cylinder_atlas_layout(
        family,
        chart_axes,
        chart_start,
        chart_end,
        node_axes,
        node_values,
        node_lower,
        node_upper,
    ):
        return None
    branch_sign = _orthogonal_cylinder_branch_sign(family, node_lower, node_upper)
    if branch_sign is None:
        return None

    # Every chart and node carries its own large-source sheet; a declared deck
    # turn between them is checked by the branch's node-binding contract.
    for chart in range(len(chart_axes)):
        first = float(chart_start[chart, theta_axis])
        last = float(chart_end[chart, theta_axis])
        if not first < last:
            return None
        graph = orthogonal_cylinder_graph_bounds(
            family, first, last, branch_sign, float(chart_start[chart, large_angle_axis])
        )
        if (
            graph is None
            or np.any(chart_lower[chart] > graph[0])
            or np.any(chart_upper[chart] < graph[1])
        ):
            return None
    node_references = (
        0.5 * node_lower[:, large_angle_axis] + 0.5 * node_upper[:, large_angle_axis]
    )
    for node, value in enumerate(node_values):
        if int(node_axes[node]) == large_angle_axis:
            if not orthogonal_cylinder_seam_node_certified(
                family, branch_sign, float(value), node_lower[node], node_upper[node]
            ):
                return None
            continue
        root = orthogonal_cylinder_graph_bounds(
            family, float(value), float(value), branch_sign, float(node_references[node])
        )
        if (
            root is None
            or np.any(node_lower[node] > root[0])
            or np.any(node_upper[node] < root[1])
        ):
            return None
    closing_root = orthogonal_cylinder_closing_root_bounds(
        family, float(node_values[0]), branch_sign, float(node_references[-1])
    )
    if (
        closing_root is None
        or np.any(node_lower[-1] > closing_root[0])
        or np.any(node_upper[-1] < closing_root[1])
    ):
        return None

    # The exact graph has |d gamma / d theta| >= small_radius > 0: the small
    # cylinder's two radial coordinates contribute speed exactly small_radius.
    if (
        family.small_radius <= 0
        or family.large_radius**2 <= (abs(family.delta_n) + family.small_radius) ** 2
    ):
        return None

    from ._intersection_kernel import (
        approximate_inverses,
        interval_magnitude,
        point_times_interval,
        round_interval,
    )

    chart_preconditioners = np.empty((len(chart_axes), 3, 3), dtype=np.float64)
    chart_contractions = np.empty((len(chart_axes),), dtype=np.float64)
    free = np.asarray(
        tuple(axis for axis in range(4) if axis != theta_axis), dtype=np.int64
    )
    identity = np.eye(3, dtype=np.float64)[None]
    for chart in range(len(chart_axes)):
        lower = chart_lower[chart : chart + 1]
        upper = chart_upper[chart : chart + 1]
        center = 0.5 * lower + 0.5 * upper
        center[0, theta_axis] = (
            0.5 * chart_start[chart, theta_axis] + 0.5 * chart_end[chart, theta_axis]
        )
        point_jacobian = prepared.point_jacobians(center)[:, :, free]
        preconditioner = approximate_inverses(point_jacobian)
        jacobian_lower, jacobian_upper = prepared.jacobian.evaluate(lower, upper)
        product_lower, product_upper = point_times_interval(
            preconditioner,
            jacobian_lower[:, :, free],
            jacobian_upper[:, :, free],
        )
        residual_lower, residual_upper = round_interval(
            identity - product_upper, identity - product_lower
        )
        contraction = np.max(
            np.nextafter(
                np.sum(interval_magnitude(residual_lower, residual_upper), axis=2)
                * (1.0 + 8.0 * np.finfo(np.float64).eps),
                np.inf,
            ),
            axis=1,
        )
        chart_preconditioners[chart] = preconditioner[0]
        chart_contractions[chart] = contraction[0]
    if not (
        np.all(np.isfinite(chart_preconditioners))
        and np.all(np.isfinite(chart_contractions))
    ):
        return None
    return chart_preconditioners, chart_contractions


# Prepared interval programs of a concrete branch's source pair, keyed by the
# branch's identity and dropped when the branch is collected. Preparation is
# pure in the immutable source regions; enclosure queries repeat it otherwise.
_BRANCH_PREPARED: dict[int, KrawczykEvaluator] = {}


def _branch_prepared_pair(curve: IntersectionCurve, /) -> KrawczykEvaluator:
    key = id(curve)
    prepared = _BRANCH_PREPARED.get(key)
    if prepared is not None:
        return prepared
    prepared = _prepare_pair(
        SurfacePairSystem(curve.first.patch, curve.second.patch),
        source_regions=(curve.first, curve.second),
    )
    if _region_is_concrete(curve.first) and _region_is_concrete(curve.second):
        _BRANCH_PREPARED[key] = prepared
        weakref.finalize(curve, _BRANCH_PREPARED.pop, key, None)
    return prepared


def _prepare_pair(
    system: SurfacePairSystem,
    /,
    *,
    source_regions: tuple[SurfaceRegion, SurfaceRegion] | None = None,
) -> KrawczykEvaluator:
    """Prepare an exact source pair, using a certified affine view when available."""
    exact = _exact_ruled_plane_pair_prepared(system.first, system.second)
    if exact is not None:
        return exact
    from ._tspline import TSplineSurfacePatch

    if isinstance(system.first, TSplineSurfacePatch) or isinstance(
        system.second, TSplineSurfacePatch
    ):
        from ._tspline_intersection import _prepare_tspline_pair

        return _prepare_tspline_pair(system, source_regions)
    revolution = _revolution_cylinder_pair_prepared(system, source_regions)
    return revolution if revolution is not None else _PreparedPair(system)


class _PreparedNodeJacobian(StrictModule):
    """Pair source Jacobian plus the exact fixed-parameter equation row."""

    system: _NodeSystem
    pair_jacobian: IntervalEvaluator

    def __init__(
        self,
        system: _NodeSystem,
        pair_jacobian: IntervalEvaluator | None = None,
        /,
    ) -> None:
        self.system = system
        self.pair_jacobian = (
            _PreparedPairJacobian(system.pair) if pair_jacobian is None else pair_jacobian
        )

    def evaluate(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        pair_lower, pair_upper = self.pair_jacobian.evaluate(lower_, upper_)
        result_lower = np.zeros((pair_lower.shape[0], 4, 4), dtype=np.float64)
        result_upper = np.zeros_like(result_lower)
        result_lower[:, :3, :] = pair_lower
        result_upper[:, :3, :] = pair_upper
        result_lower[:, 3, self.system.axis] = 1.0
        result_upper[:, 3, self.system.axis] = 1.0
        return result_lower, result_upper


class _PreparedNodeValue:
    def __init__(
        self,
        pair_value: IntervalEvaluator,
        axis: int,
        value: float,
        /,
    ) -> None:
        self.pair_value = pair_value
        self.axis = axis
        self.value = value

    def evaluate(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import interval_subtract

        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        pair_lower, pair_upper = self.pair_value.evaluate(lower_, upper_)
        fixed = np.asarray(self.value, dtype=np.float64)
        fixed_lower, fixed_upper = interval_subtract(
            (lower_[:, self.axis], upper_[:, self.axis]), (fixed, fixed)
        )
        fixed_lower = np.asarray(fixed_lower, dtype=np.float64).reshape(
            (pair_lower.shape[0], 1)
        )
        fixed_upper = np.asarray(fixed_upper, dtype=np.float64).reshape(
            (pair_upper.shape[0], 1)
        )
        return (
            np.concatenate((pair_lower, fixed_lower), axis=1),
            np.concatenate((pair_upper, fixed_upper), axis=1),
        )


class _PreparedNode(StrictModule):
    system: _NodeSystem
    pair_prepared: KrawczykEvaluator | None
    value: IntervalEvaluator
    jacobian: _PreparedNodeJacobian

    def __init__(
        self,
        system: _NodeSystem,
        pair_prepared: KrawczykEvaluator | None = None,
        /,
        *,
        source_regions: tuple[SurfaceRegion, SurfaceRegion] | None = None,
    ) -> None:
        if pair_prepared is not None:
            prepared_system = getattr(pair_prepared, "system", None)
            if not isinstance(prepared_system, SurfacePairSystem):
                raise TypeError(
                    "Prepared node bounds require a source-bound surface-pair preparation."
                )
            if (
                prepared_system.first is not system.pair.first
                or prepared_system.second is not system.pair.second
            ):
                raise ValueError(
                    "Prepared node bounds must bind the exact surface-pair sources."
                )
            prepared_regions = getattr(pair_prepared, "source_regions", None)
            if prepared_regions is not None and (
                source_regions is None
                or any(
                    prepared is not expected
                    for prepared, expected in zip(
                        prepared_regions,
                        source_regions,
                        strict=True,
                    )
                )
            ):
                raise ValueError(
                    "Prepared node bounds must bind the exact source regions."
                )
        self.system = system
        self.pair_prepared = pair_prepared
        self.value = (
            prepare_interval_function(system.residual, 4)
            if pair_prepared is None
            else _PreparedNodeValue(pair_prepared.value, system.axis, system.value)
        )
        self.jacobian = _PreparedNodeJacobian(
            system,
            None if pair_prepared is None else pair_prepared.jacobian,
        )

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        if self.pair_prepared is not None:
            values = np.asarray(points, dtype=np.float64)
            pair = np.asarray(
                self.pair_prepared.point_jacobians(values), dtype=np.float64
            )
            row = np.zeros((pair.shape[0], 1, 4), dtype=np.float64)
            row[:, 0, self.system.axis] = 1.0
            return np.concatenate((pair, row), axis=1)
        values = jnp.asarray(points, dtype=jnp.float64)
        return np.asarray(
            _batched_node_jacobians(
                self.system.pair.first,
                self.system.pair.second,
                values,
                jnp.asarray(self.system.axis, dtype=jnp.int32),
            ),
            dtype=np.float64,
        )


class _ExactAffinePairValue:
    def __init__(self, prepared: _ExactAffinePairPrepared, /) -> None:
        self.prepared = prepared

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._rational import fraction_down, fraction_up

        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        if lower_.ndim != 2 or lower_.shape[1] != 4 or upper_.shape != lower_.shape:
            raise ValueError("Affine pair boxes must have aligned shape (count, 4).")
        result_lower = np.empty((lower_.shape[0], 3), dtype=np.float64)
        result_upper = np.empty_like(result_lower)
        for row in range(lower_.shape[0]):
            parameters = tuple(
                (
                    Fraction.from_float(float(lower_[row, axis])),
                    Fraction.from_float(float(upper_[row, axis])),
                )
                for axis in range(4)
            )
            for component in range(3):
                value_lower, value_upper = self.prepared.origin_bounds[component]
                for axis in range(4):
                    coefficient_lower, coefficient_upper = self.prepared.direction_bounds[
                        component
                    ][axis]
                    parameter_lower, parameter_upper = parameters[axis]
                    products = (
                        coefficient_lower * parameter_lower,
                        coefficient_lower * parameter_upper,
                        coefficient_upper * parameter_lower,
                        coefficient_upper * parameter_upper,
                    )
                    value_lower += min(products)
                    value_upper += max(products)
                result_lower[row, component] = fraction_down(value_lower)
                result_upper[row, component] = fraction_up(value_upper)
        return result_lower, result_upper


class _ExactAffinePairJacobian:
    def __init__(self, prepared: _ExactAffinePairPrepared, /) -> None:
        self.prepared = prepared

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        count = np.asarray(lower).shape[0]
        if np.asarray(lower).shape != (count, 4) or np.asarray(upper).shape != (
            count,
            4,
        ):
            raise ValueError("Affine pair Jacobian boxes must have shape (count, 4).")
        return (
            np.broadcast_to(self.prepared.jacobian_lower, (count, 3, 4)),
            np.broadcast_to(self.prepared.jacobian_upper, (count, 3, 4)),
        )


class _ExactAffinePairPrepared:
    """Exact interval Krawczyk evaluator for an affine ruled/plane pair."""

    def __init__(
        self,
        origin_bounds: tuple[
            tuple[Fraction, Fraction],
            tuple[Fraction, Fraction],
            tuple[Fraction, Fraction],
        ],
        direction_bounds: tuple[
            tuple[
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
            ],
            tuple[
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
            ],
            tuple[
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
                tuple[Fraction, Fraction],
            ],
        ],
        /,
    ) -> None:
        from .._rational import fraction_down, fraction_up

        if (
            len(origin_bounds) != 3
            or len(direction_bounds) != 3
            or any(len(row) != 4 for row in direction_bounds)
            or any(lower > upper for lower, upper in origin_bounds)
            or any(lower > upper for row in direction_bounds for lower, upper in row)
        ):
            raise ValueError(
                "Affine source coefficient intervals must be ordered 3x4 data."
            )
        self.origin_bounds = origin_bounds
        self.direction_bounds = direction_bounds
        self.jacobian_lower = np.asarray(
            tuple(
                tuple(fraction_down(lower) for lower, _ in row)
                for row in direction_bounds
            ),
            dtype=np.float64,
        )
        self.jacobian_upper = np.asarray(
            tuple(
                tuple(fraction_up(upper) for _, upper in row) for row in direction_bounds
            ),
            dtype=np.float64,
        )
        self.point_jacobian = np.asarray(
            tuple(
                tuple(float((lower + upper) / 2) for lower, upper in row)
                for row in direction_bounds
            ),
            dtype=np.float64,
        )
        if not (
            np.all(np.isfinite(self.jacobian_lower))
            and np.all(np.isfinite(self.jacobian_upper))
            and np.all(np.isfinite(self.point_jacobian))
        ):
            raise ValueError("Exact affine pair coefficients need finite shadows.")
        self.value = _ExactAffinePairValue(self)
        self.jacobian = _ExactAffinePairJacobian(self)

    def intersect_exact_coordinates(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> None:
        """Intersect exact one-coordinate source equations into owned bounds."""
        from .._rational import fraction_down, fraction_up

        for origin, row in zip(self.origin_bounds, self.direction_bounds, strict=True):
            if origin[0] != origin[1] or any(first != last for first, last in row):
                continue
            active = tuple(axis for axis, (value, _) in enumerate(row) if value != 0)
            if len(active) != 1:
                continue
            axis = active[0]
            value = -origin[0] / row[axis][0]
            lower[axis] = max(lower[axis], fraction_down(value))
            upper[axis] = min(upper[axis], fraction_up(value))
            if lower[axis] > upper[axis]:
                raise ValueError(
                    "The source graph bounds contradict an exact coordinate equation."
                )

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        count = np.asarray(points).shape[0]
        return np.broadcast_to(self.point_jacobian, (count, 3, 4))


class _ExactAffineNodeValue:
    def __init__(
        self, pair: _ExactAffinePairPrepared, axis: int, value: float, /
    ) -> None:
        self.pair, self.axis, self.value = pair, axis, value

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._rational import fraction_down, fraction_up

        lower_, upper_ = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        pair_lower, pair_upper = self.pair.value.evaluate(lower_, upper_)
        fixed = Fraction.from_float(float(self.value))
        node_lower = np.empty((lower_.shape[0], 1), dtype=np.float64)
        node_upper = np.empty_like(node_lower)
        for row in range(lower_.shape[0]):
            node_lower[row, 0] = fraction_down(
                Fraction.from_float(float(lower_[row, self.axis])) - fixed
            )
            node_upper[row, 0] = fraction_up(
                Fraction.from_float(float(upper_[row, self.axis])) - fixed
            )
        return (
            np.concatenate((pair_lower, node_lower), axis=1),
            np.concatenate((pair_upper, node_upper), axis=1),
        )


class _ExactAffineNodeJacobian:
    def __init__(self, pair: _ExactAffinePairPrepared, axis: int, /) -> None:
        self.pair, self.axis = pair, axis

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_, upper_ = np.asarray(lower), np.asarray(upper)
        pair_lower, pair_upper = self.pair.jacobian.evaluate(lower_, upper_)
        node_row = np.zeros((lower_.shape[0], 1, 4), dtype=np.float64)
        node_row[:, 0, self.axis] = 1.0
        return (
            np.concatenate((pair_lower, node_row), axis=1),
            np.concatenate((pair_upper, node_row), axis=1),
        )


class _ExactAffineNodePrepared:
    def __init__(
        self, pair: _ExactAffinePairPrepared, axis: int, value: float, /
    ) -> None:
        self.pair = pair
        self.value = _ExactAffineNodeValue(pair, axis, value)
        self.jacobian = _ExactAffineNodeJacobian(pair, axis)

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        pair = self.pair.point_jacobians(points)
        node_row = np.zeros((pair.shape[0], 1, 4), dtype=np.float64)
        node_row[:, 0, self.jacobian.axis] = 1.0
        return np.concatenate((pair, node_row), axis=1)


def _exact_ruled_plane_pair_prepared(
    first: AbstractSurfacePatch,
    second: AbstractSurfacePatch,
    /,
) -> _ExactAffinePairPrepared | None:
    from ...discretization._exact_algebraic_geometry import coefficient_interval
    from .._rational import rational_vector
    from ._boolean_core import _pcurve_execution_shadows_match
    from ._patches import (
        CircleCurve,
        ConicRootLinePCurve,
        CylinderPatch,
        PlanePatch,
        RuledSurface,
        SurfaceMappedCurve,
    )

    if isinstance(first, RuledSurface) and isinstance(second, PlanePatch):
        ruled, plane, ruled_first = first, second, True
    elif isinstance(first, PlanePatch) and isinstance(second, RuledSurface):
        ruled, plane, ruled_first = second, first, False
    else:
        return None

    def add(
        left: tuple[Fraction, Fraction],
        right: tuple[Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction]:
        return left[0] + right[0], left[1] + right[1]

    def subtract(
        left: tuple[Fraction, Fraction],
        right: tuple[Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction]:
        return left[0] - right[1], left[1] - right[0]

    def scale(
        interval: tuple[Fraction, Fraction], value: Fraction, /
    ) -> tuple[Fraction, Fraction]:
        values = interval[0] * value, interval[1] * value
        return min(values), max(values)

    def coefficient_bounds(value: ExactScalar, /) -> tuple[Fraction, Fraction]:
        return coefficient_interval(value, bits=128)

    def rail_data(
        rail: AbstractCurve, /
    ) -> (
        tuple[
            tuple[tuple[Fraction, Fraction], ...],
            tuple[ExactScalar, ExactScalar, ExactScalar],
        ]
        | None
    ):
        if (
            not isinstance(rail, SurfaceMappedCurve)
            or not isinstance(rail.surface, CylinderPatch)
            or not isinstance(rail.pcurve, ConicRootLinePCurve)
            or not _pcurve_execution_shadows_match(rail.pcurve)
        ):
            return None
        patch, pcurve = rail.surface, rail.pcurve
        conic = pcurve.root.conic
        if not isinstance(conic, CircleCurve) or (
            conic.center_exact != patch.origin_exact
            or conic.first_axis_exact != patch.first_axis_exact
            or conic.second_axis_exact != patch.second_axis_exact
            or conic.radius_exact != patch.radius_exact
        ):
            return None
        domain = rail.parameter_domain
        if domain is None or not np.isfinite(domain[0]):
            return None
        parameter = float(domain[0])
        try:
            point_box = np.asarray(
                rail.bounding_box(parameter, parameter), dtype=np.float64
            )
        except (NotImplementedError, ValueError):
            return None
        if (
            point_box.shape != (2, 3)
            or not np.all(np.isfinite(point_box))
            or np.any(point_box[0] > point_box[1])
        ):
            return None
        point_bounds = (
            (
                Fraction.from_float(float(point_box[0, 0])),
                Fraction.from_float(float(point_box[1, 0])),
            ),
            (
                Fraction.from_float(float(point_box[0, 1])),
                Fraction.from_float(float(point_box[1, 1])),
            ),
            (
                Fraction.from_float(float(point_box[0, 2])),
                Fraction.from_float(float(point_box[1, 2])),
            ),
        )
        direction: tuple[ExactScalar, ExactScalar, ExactScalar] = (
            pcurve.height_direction_exact * patch.axis_exact[0],
            pcurve.height_direction_exact * patch.axis_exact[1],
            pcurve.height_direction_exact * patch.axis_exact[2],
        )
        direction_bounds = tuple(coefficient_bounds(value) for value in direction)
        parameter_exact = Fraction.from_float(parameter)
        origin_bounds = (
            subtract(
                point_bounds[0],
                scale(direction_bounds[0], parameter_exact),
            ),
            subtract(
                point_bounds[1],
                scale(direction_bounds[1], parameter_exact),
            ),
            subtract(
                point_bounds[2],
                scale(direction_bounds[2], parameter_exact),
            ),
        )
        origin_bounds = tuple(
            coefficient_bounds(
                patch.origin_exact[axis]
                + pcurve.height_origin_exact * patch.axis_exact[axis]
            )
            if patch.first_axis_exact[axis] == 0 and patch.second_axis_exact[axis] == 0
            else origin_bounds[axis]
            for axis in range(3)
        )
        return origin_bounds, direction

    rails = (ruled.first, ruled.second)
    first_data, second_data = (rail_data(rail) for rail in rails)
    if first_data is None or second_data is None:
        return None
    first_origin, first_direction = first_data
    second_origin, second_direction = second_data
    if first_direction != second_direction:
        return None
    plane_origin = rational_vector(np.asarray(plane.origin))
    plane_first = rational_vector(np.asarray(plane.first_axis))
    plane_second = rational_vector(np.asarray(plane.second_axis))
    if any(len(vector) != 3 for vector in (plane_origin, plane_first, plane_second)):
        return None
    plane_origin_bounds = tuple(coefficient_bounds(value) for value in plane_origin)
    plane_first_bounds = tuple(coefficient_bounds(value) for value in plane_first)
    plane_second_bounds = tuple(coefficient_bounds(value) for value in plane_second)
    direction_bounds = tuple(coefficient_bounds(value) for value in first_direction)
    rail_delta = tuple(
        subtract(second_origin[axis], first_origin[axis]) for axis in range(3)
    )
    if ruled_first:
        origin_bounds = (
            subtract(first_origin[0], plane_origin_bounds[0]),
            subtract(first_origin[1], plane_origin_bounds[1]),
            subtract(first_origin[2], plane_origin_bounds[2]),
        )
        columns = (
            direction_bounds,
            rail_delta,
            tuple((-upper, -lower) for lower, upper in plane_first_bounds),
            tuple((-upper, -lower) for lower, upper in plane_second_bounds),
        )
    else:
        origin_bounds = (
            subtract(plane_origin_bounds[0], first_origin[0]),
            subtract(plane_origin_bounds[1], first_origin[1]),
            subtract(plane_origin_bounds[2], first_origin[2]),
        )
        columns = (
            plane_first_bounds,
            plane_second_bounds,
            tuple((-upper, -lower) for lower, upper in direction_bounds),
            tuple((-upper, -lower) for lower, upper in rail_delta),
        )
    rows = (
        (columns[0][0], columns[1][0], columns[2][0], columns[3][0]),
        (columns[0][1], columns[1][1], columns[2][1], columns[3][1]),
        (columns[0][2], columns[1][2], columns[2][2], columns[3][2]),
    )
    return _ExactAffinePairPrepared(origin_bounds, rows)


def _exact_ruled_plane_node_prepared(
    first: AbstractSurfacePatch,
    second: AbstractSurfacePatch,
    axis: int,
    value: float,
    /,
) -> _ExactAffineNodePrepared | None:
    pair = _exact_ruled_plane_pair_prepared(first, second)
    return None if pair is None else _ExactAffineNodePrepared(pair, axis, value)


class IntersectionCurvePoint(StrictModule):
    """One coupled JAX realization of a spatial branch and both source charts."""

    point: Array
    first_parameters: Array
    second_parameters: Array
    tangent: Array
    parameter_bound: Array
    gap: Array
    chart: Array

    def __init__(
        self,
        point: Array,
        first_parameters: Array,
        second_parameters: Array,
        tangent: Array,
        parameter_bound: Array,
        gap: Array,
        chart: Array,
    ) -> None:
        self.point = point
        self.first_parameters = first_parameters
        self.second_parameters = second_parameters
        self.tangent = tangent
        self.parameter_bound = parameter_bound
        self.gap = gap
        self.chart = chart


@jax.jit
def _curve_residual(
    first: AbstractSurfacePatch, second: AbstractSurfacePatch, q: Array
) -> Array:
    return first.evaluate(q[:2]) - second.evaluate(q[2:])


@jax.jit
def _curve_jacobian(
    first: AbstractSurfacePatch, second: AbstractSurfacePatch, q: Array
) -> Array:
    return jax.jacfwd(_curve_residual, argnums=2)(first, second, q)


@jax.jit
def _batched_curve_jacobians(
    first: AbstractSurfacePatch,
    second: AbstractSurfacePatch,
    points: Array,
) -> Array:
    return jax.vmap(_curve_jacobian, in_axes=(None, None, 0))(first, second, points)


@jax.jit
def _batched_node_jacobians(
    first: AbstractSurfacePatch,
    second: AbstractSurfacePatch,
    points: Array,
    axis: Array,
) -> Array:
    pair = _batched_curve_jacobians(first, second, points)
    fixed = jnp.zeros((points.shape[0], 1, 4), dtype=points.dtype)
    fixed = fixed.at[:, 0, axis].set(1.0)
    return jnp.concatenate((pair, fixed), axis=1)


def _evaluate_orthogonal_cylinder_curve_point(
    curve: IntersectionCurve,
    parameter: Array,
    large_index: int,
    /,
) -> tuple[Array, ...]:
    """Evaluate the exact line-revolution branch and its two analytic p-curves."""
    from ._patches import ExactLineCurve, LineCurve, RevolutionSurface

    if large_index == 0:
        large_surface, small_surface = curve.first.patch, curve.second.patch
    else:
        large_surface, small_surface = curve.second.patch, curve.first.patch
    if not isinstance(large_surface, RevolutionSurface) or not isinstance(
        small_surface, RevolutionSurface
    ):
        raise TypeError("The orthogonal-cylinder evaluator requires revolution sources.")

    def profile_data(
        surface: RevolutionSurface, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        profile = surface.curve
        if isinstance(profile, LineCurve):
            origin, direction = profile.origin, profile.direction
        elif isinstance(profile, ExactLineCurve):
            origin, direction = profile.first, profile.last - profile.first
        else:
            raise TypeError("The orthogonal-cylinder evaluator requires line profiles.")
        axis_origin = surface.axis_origin
        axis = surface.axis_direction
        relative = origin - axis_origin
        axial_origin = jnp.sum(relative * axis)
        radial = relative - axial_origin * axis
        radius = jnp.linalg.norm(radial)
        first_axis = radial / radius
        second_axis = jnp.cross(axis, first_axis)
        center = axis_origin + axial_origin * axis
        slope = jnp.sum(direction * axis)
        return center, axis, first_axis, second_axis, radius, slope

    large_center, large_axis, large_first, large_second, large_radius, large_slope = (
        profile_data(large_surface)
    )
    small_center, small_axis, small_first, small_second, small_radius, small_slope = (
        profile_data(small_surface)
    )
    small_index = 1 - large_index
    theta_axis = 2 * small_index
    small_profile_axis = theta_axis + 1
    large_angle_axis = 2 * large_index
    last_chart = curve.chart_axes.shape[0] - 1
    chart = jnp.minimum(jnp.floor(parameter).astype(jnp.int32), last_chart)
    start, end = curve.chart_start[chart], curve.chart_end[chart]
    fraction = parameter - chart.astype(jnp.float64)
    theta_start, theta_end = start[theta_axis], end[theta_axis]
    theta = theta_start + fraction * (theta_end - theta_start)
    normal = jnp.cross(large_axis, small_axis)
    displacement = small_center - large_center
    delta_n = jnp.sum(displacement * normal)
    delta_large_axis = jnp.sum(displacement * large_axis)
    delta_small_axis = jnp.sum(displacement * small_axis)
    start_x = delta_small_axis + small_slope * start[small_profile_axis]
    branch_sign = jnp.where(start_x >= 0.0, 1.0, -1.0)

    def parameters_at_angle(angle: Array, /) -> Array:
        radial_small = small_radius * (
            jnp.cos(angle) * small_first + jnp.sin(angle) * small_second
        )
        radial_n = delta_n + jnp.sum(radial_small * normal)
        axial_large = delta_large_axis + jnp.sum(radial_small * large_axis)
        axial_large_coordinate = branch_sign * jnp.sqrt(
            large_radius * large_radius - radial_n * radial_n
        )
        world = (
            large_center
            + axial_large_coordinate * small_axis
            + radial_n * normal
            + axial_large * large_axis
        )
        radial_large = world - large_center - axial_large * large_axis
        angle_large = jnp.arctan2(
            jnp.sum(radial_large * large_second),
            jnp.sum(radial_large * large_first),
        )
        period = 2.0 * jnp.pi
        angle_large += period * jnp.round(
            (start[large_angle_axis] - angle_large) / period
        )
        large_profile = axial_large / large_slope
        small_profile = (axial_large_coordinate - delta_small_axis) / small_slope
        if large_index == 0:
            return jnp.stack((angle_large, large_profile, angle, small_profile))
        return jnp.stack((angle, small_profile, angle_large, large_profile))

    parameters = parameters_at_angle(theta)
    parameter_tangent = jax.jacfwd(parameters_at_angle)(theta) * (theta_end - theta_start)
    first = curve.first.patch.evaluate(parameters[:2])
    second = curve.second.patch.evaluate(parameters[2:])
    first_jacobian = jax.jacfwd(curve.first.patch.evaluate)(parameters[:2])
    tangent = first_jacobian @ parameter_tangent[:2]
    tangent_norm = jnp.linalg.norm(tangent)
    tangent = jnp.where(
        tangent_norm > 0.0, tangent / tangent_norm, jnp.zeros_like(tangent)
    )
    radius = jnp.maximum(
        jnp.abs(curve.chart_upper[chart] - parameters),
        jnp.abs(parameters - curve.chart_lower[chart]),
    )
    return (
        first,
        parameters[:2],
        parameters[2:],
        tangent,
        jnp.max(radius),
        jnp.linalg.norm(first - second),
        chart,
    )


@jax.jit
def _evaluate_orthogonal_cylinder_curve_point_large_first(
    curve: IntersectionCurve, parameter: Array, /
) -> tuple[Array, ...]:
    return _evaluate_orthogonal_cylinder_curve_point(curve, parameter, 0)


@jax.jit
def _evaluate_orthogonal_cylinder_curve_point_large_second(
    curve: IntersectionCurve, parameter: Array, /
) -> tuple[Array, ...]:
    return _evaluate_orthogonal_cylinder_curve_point(curve, parameter, 1)


@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7))
def _evaluate_orthogonal_cylinder_parameters_on_chart(
    curve: IntersectionCurve,
    chart: int,
    parameter: Array,
    large_index: int,
    branch_sign: int,
    angle_mode: int,
    angle_sign: int,
    angle_turns: int,
    /,
) -> Array:
    """Evaluate the exact source p-curve on one certified cylinder chart."""
    from ._patches import ExactLineCurve, LineCurve, RevolutionSurface

    large_surface, small_surface = (
        (curve.first.patch, curve.second.patch)
        if large_index == 0
        else (curve.second.patch, curve.first.patch)
    )
    if not isinstance(large_surface, RevolutionSurface) or not isinstance(
        small_surface, RevolutionSurface
    ):
        raise TypeError("The orthogonal-cylinder evaluator requires revolution sources.")

    def profile_data(
        surface: RevolutionSurface, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        profile = surface.curve
        if isinstance(profile, LineCurve):
            origin, direction = profile.origin, profile.direction
        elif isinstance(profile, ExactLineCurve):
            origin, direction = profile.first, profile.last - profile.first
        else:
            raise TypeError("The orthogonal-cylinder evaluator requires line profiles.")
        axis_origin, axis = surface.axis_origin, surface.axis_direction
        relative = origin - axis_origin
        axial_origin = jnp.sum(relative * axis)
        radial = relative - axial_origin * axis
        radius = jnp.linalg.norm(radial)
        first_axis = radial / radius
        second_axis = jnp.cross(axis, first_axis)
        center = axis_origin + axial_origin * axis
        slope = jnp.sum(direction * axis)
        return center, axis, first_axis, second_axis, radius, slope

    large_center, large_axis, large_first, large_second, large_radius, large_slope = (
        profile_data(large_surface)
    )
    small_center, small_axis, small_first, small_second, small_radius, small_slope = (
        profile_data(small_surface)
    )
    small_index = 1 - large_index
    theta_axis = 2 * small_index
    start = curve.chart_start[chart]
    end = curve.chart_end[chart]
    fraction = parameter - chart
    theta = start[theta_axis] + fraction * (end[theta_axis] - start[theta_axis])
    normal = jnp.cross(large_axis, small_axis)
    displacement = small_center - large_center
    delta_n = jnp.sum(displacement * normal)
    delta_large_axis = jnp.sum(displacement * large_axis)
    delta_small_axis = jnp.sum(displacement * small_axis)
    radial_small = small_radius * (
        jnp.cos(theta) * small_first + jnp.sin(theta) * small_second
    )
    radial_n = delta_n + jnp.sum(radial_small * normal)
    axial_large = delta_large_axis + jnp.sum(radial_small * large_axis)
    axial_small = branch_sign * jnp.sqrt(
        large_radius * large_radius - radial_n * radial_n
    )
    first_component = axial_small * jnp.sum(
        small_axis * large_first
    ) + radial_n * jnp.sum(normal * large_first)
    second_component = axial_small * jnp.sum(
        small_axis * large_second
    ) + radial_n * jnp.sum(normal * large_second)
    if angle_mode == 1:
        angle = angle_sign * 0.5 * jnp.pi - jnp.arctan(first_component / second_component)
    else:
        angle = jnp.arctan(second_component / first_component)
        if angle_sign < 0:
            angle += jnp.pi
    angle += angle_turns * 2.0 * jnp.pi
    large_profile = axial_large / large_slope
    small_profile = (axial_small - delta_small_axis) / small_slope
    if large_index == 0:
        return jnp.stack((angle, large_profile, theta, small_profile))
    return jnp.stack((theta, small_profile, angle, large_profile))


def _implicit_chart_derivative_bounds(
    curve: IntersectionCurve, chart: int, first: float, last: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose all four chart-parameter derivatives on one certified chart.

    The chart axis moves affinely with the branch parameter. The free
    parameters follow the implicit function theorem on the certified chart
    boxes: ``J_free dq_free = -J_axis dq_axis`` with surface Jacobian
    enclosures, solved by a preconditioned interval contraction.
    """
    from .._interval_enclosure import interval_add, interval_multiply, interval_subtract
    from .._rational import fraction_down, fraction_up

    axis = int(np.asarray(curve.chart_axes)[chart])
    free = tuple(index for index in range(4) if index != axis)
    start = float(np.asarray(curve.chart_start)[chart, axis])
    end = float(np.asarray(curve.chart_end)[chart, axis])
    rate_exact = Fraction.from_float(end) - Fraction.from_float(start)
    rate = (
        np.asarray(fraction_down(rate_exact), dtype=np.float64),
        np.asarray(fraction_up(rate_exact), dtype=np.float64),
    )
    boxes = np.asarray(curve.parameter_enclosures(first, last), dtype=np.float64)
    box_lower = np.min(boxes[:, 0, :], axis=0)
    box_upper = np.max(boxes[:, 1, :], axis=0)
    first_lower, first_upper = curve.first.patch.derivative_bounds(
        np.stack((box_lower[:2], box_upper[:2]))
    )
    second_lower, second_upper = curve.second.patch.derivative_bounds(
        np.stack((box_lower[2:], box_upper[2:]))
    )
    # Residual S1(q[:2]) - S2(q[2:]) has Jacobian [J1, -J2].
    jacobian_lower = np.concatenate(
        (np.asarray(first_lower), -np.asarray(second_upper)), axis=1
    )
    jacobian_upper = np.concatenate(
        (np.asarray(first_upper), -np.asarray(second_lower)), axis=1
    )
    if not np.all(np.isfinite(jacobian_lower)) or not np.all(np.isfinite(jacobian_upper)):
        raise ValueError("IntersectionPCurve derivative needs finite surface Jacobians.")
    matrix = (jacobian_lower[:, free], jacobian_upper[:, free])
    try:
        preconditioner = np.linalg.inv(0.5 * (matrix[0] + matrix[1]))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "IntersectionPCurve derivative interval is unresolved at a singular chart."
        ) from error
    if not np.all(np.isfinite(preconditioner)):
        raise ValueError(
            "IntersectionPCurve derivative interval is unresolved at a singular chart."
        )
    rhs = interval_multiply((-jacobian_upper[:, axis], -jacobian_lower[:, axis]), rate)

    def precondition(
        values: tuple[np.ndarray, np.ndarray], /
    ) -> tuple[np.ndarray, np.ndarray]:
        rows_lower, rows_upper = [], []
        for row in range(3):
            total = (np.zeros(values[0].shape[1:]), np.zeros(values[0].shape[1:]))
            for column in range(3):
                coefficient = preconditioner[row, column]
                total = interval_add(
                    total,
                    interval_multiply(
                        (np.asarray(coefficient), np.asarray(coefficient)),
                        (values[0][column], values[1][column]),
                    ),
                )
            rows_lower.append(np.asarray(total[0], dtype=np.float64))
            rows_upper.append(np.asarray(total[1], dtype=np.float64))
        return np.stack(rows_lower), np.stack(rows_upper)

    product_lower, product_upper = precondition(matrix)
    identity = np.eye(3)
    error_lower, error_upper = interval_subtract(
        (identity, identity), (product_lower, product_upper)
    )
    contraction = Fraction(0)
    for row in range(3):
        contraction = max(
            contraction,
            sum(
                (
                    Fraction.from_float(
                        float(
                            max(
                                abs(error_lower[row, column]),
                                abs(error_upper[row, column]),
                            )
                        )
                    )
                    for column in range(3)
                ),
                Fraction(0),
            ),
        )
    if contraction >= 1:
        raise ValueError(
            "IntersectionPCurve derivative interval is unresolved; the chart Jacobian does not contract."
        )
    center_lower, center_upper = precondition(rhs)
    radius = fraction_up(
        max(
            Fraction.from_float(float(max(abs(low), abs(high))))
            for low, high in zip(center_lower, center_upper, strict=True)
        )
        / (1 - contraction)
    )
    ball = (np.asarray(-radius), np.asarray(radius))
    free_lower, free_upper = [], []
    for row in range(3):
        total = (np.asarray(center_lower[row]), np.asarray(center_upper[row]))
        for column in range(3):
            total = interval_add(
                total,
                interval_multiply(
                    (
                        np.asarray(error_lower[row, column]),
                        np.asarray(error_upper[row, column]),
                    ),
                    ball,
                ),
            )
        free_lower.append(float(total[0]))
        free_upper.append(float(total[1]))
    lower = np.empty(4, dtype=np.float64)
    upper = np.empty(4, dtype=np.float64)
    lower[axis], upper[axis] = float(rate[0]), float(rate[1])
    lower[list(free)] = free_lower
    upper[list(free)] = free_upper
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("IntersectionPCurve derivative interval is unresolved.")
    return lower, upper


def _evaluate_curve_point_on_chart_impl(
    curve: IntersectionCurve, chart: Array, parameter: Array
) -> tuple[Array, ...]:
    fraction = parameter - chart.astype(jnp.float64)
    axis = curve.chart_axes[chart]
    free = _FREE_AXES[axis]
    start = curve.chart_start[chart]
    end = curve.chart_end[chart]
    initial = start + fraction * (end - start)
    axis_value = initial[axis]
    free_values = initial[free]
    box_lower = curve.chart_lower[chart, free]
    box_upper = curve.chart_upper[chart, free]
    preconditioner = curve.preconditioners[chart]

    def iteration(_: int, values: Array) -> Array:
        q = (
            jnp.zeros((4,), dtype=jnp.float64)
            .at[axis]
            .set(axis_value)
            .at[free]
            .set(values)
        )
        residual = _curve_residual(curve.first.patch, curve.second.patch, q)
        correction = preconditioner @ residual
        return jnp.clip(values - correction, box_lower, box_upper)

    solved = jax.lax.fori_loop(0, 64, iteration, free_values)
    q = jnp.zeros((4,), dtype=jnp.float64).at[axis].set(axis_value).at[free].set(solved)
    jacobian = _curve_jacobian(curve.first.patch, curve.second.patch, q)[:, free]
    axis_column = _curve_jacobian(curve.first.patch, curve.second.patch, q)[:, axis]
    dq_free = -jnp.linalg.solve(jacobian, axis_column) * (end[axis] - start[axis])
    dq = (
        jnp.zeros((4,), dtype=jnp.float64)
        .at[axis]
        .set(end[axis] - start[axis])
        .at[free]
        .set(dq_free)
    )
    first = curve.first.patch.evaluate(q[:2])
    second = curve.second.patch.evaluate(q[2:])
    first_jacobian = jax.jacfwd(curve.first.patch.evaluate)(q[:2])
    tangent = first_jacobian @ dq[:2]
    tangent_norm = jnp.linalg.norm(tangent)
    tangent = jnp.where(
        tangent_norm > 0.0, tangent / tangent_norm, jnp.zeros_like(tangent)
    )
    radius = jnp.maximum(
        jnp.abs(curve.chart_upper[chart] - q), jnp.abs(q - curve.chart_lower[chart])
    )
    parameter_bound = jnp.max(radius)
    return (
        first,
        q[:2],
        q[2:],
        tangent,
        parameter_bound,
        jnp.linalg.norm(first - second),
        chart,
    )


@jax.jit
def _evaluate_curve_point_on_chart(
    curve: IntersectionCurve, chart: Array, parameter: Array
) -> tuple[Array, ...]:
    return _evaluate_curve_point_on_chart_impl(curve, chart, parameter)


@partial(jax.jit, static_argnums=(1, 2))
def _evaluate_curve_point_on_fixed_chart(
    curve: IntersectionCurve, chart: int, axis: int, parameter: Array
) -> tuple[Array, ...]:
    """Evaluate one fixed source chart without dynamic parameter indexing."""
    free = tuple(index for index in range(4) if index != axis)
    start, end = curve.chart_start[chart], curve.chart_end[chart]
    fraction = parameter - chart
    initial = start + fraction * (end - start)
    axis_value = initial[axis]
    free_values = jnp.stack(tuple(initial[index] for index in free))
    box_lower = jnp.stack(tuple(curve.chart_lower[chart, index] for index in free))
    box_upper = jnp.stack(tuple(curve.chart_upper[chart, index] for index in free))
    preconditioner = curve.preconditioners[chart]

    def coordinates(fixed_value: Array, free_value: Array) -> Array:
        return jnp.stack(
            tuple(
                fixed_value if index == axis else free_value[free.index(index)]
                for index in range(4)
            )
        )

    def iteration(_: int, values: Array) -> Array:
        q = coordinates(axis_value, values)
        residual = _curve_residual(curve.first.patch, curve.second.patch, q)
        correction = preconditioner @ residual
        return jnp.clip(values - correction, box_lower, box_upper)

    solved = jax.lax.fori_loop(0, 64, iteration, free_values)
    q = coordinates(axis_value, solved)
    full_jacobian = _curve_jacobian(curve.first.patch, curve.second.patch, q)
    jacobian = full_jacobian[:, free]
    axis_column = full_jacobian[:, axis]
    dq_free = -jnp.linalg.solve(jacobian, axis_column) * (end[axis] - start[axis])
    dq = coordinates(end[axis] - start[axis], dq_free)
    first = curve.first.patch.evaluate(q[:2])
    second = curve.second.patch.evaluate(q[2:])
    first_jacobian = jax.jacfwd(curve.first.patch.evaluate)(q[:2])
    tangent = first_jacobian @ dq[:2]
    tangent_norm = jnp.linalg.norm(tangent)
    tangent = jnp.where(
        tangent_norm > 0.0, tangent / tangent_norm, jnp.zeros_like(tangent)
    )
    radius = jnp.maximum(
        jnp.abs(curve.chart_upper[chart] - q), jnp.abs(q - curve.chart_lower[chart])
    )
    return (
        first,
        q[:2],
        q[2:],
        tangent,
        jnp.max(radius),
        jnp.linalg.norm(first - second),
        jnp.asarray(chart, dtype=jnp.int32),
    )


@jax.jit
def _evaluate_curve_point(
    curve: IntersectionCurve, parameter: Array
) -> tuple[Array, ...]:
    last_chart = curve.chart_axes.shape[0] - 1
    chart = jnp.minimum(jnp.floor(parameter).astype(jnp.int32), last_chart)
    return _evaluate_curve_point_on_chart(curve, chart, parameter)


class IntersectionCurve(AbstractCurve):
    """One certified branch of two source surfaces, never a fitted point cloud.

    Each chart proves a unique graph of the coupled surface-pair zero set over
    one parameter coordinate. Consecutive chart endpoints are separately
    isolated source roots, so the atlas is a connected branch definition.
    """

    first: SurfaceRegion
    second: SurfaceRegion
    chart_axes: Array
    chart_start: Array
    chart_end: Array
    chart_lower: Array
    chart_upper: Array
    preconditioners: Array
    contractions: Array
    node_axes: Array
    node_values: Array
    node_lower: Array
    node_upper: Array
    node_period_shifts: Array
    closed: bool = eqx.field(static=True)
    start_kind: IntersectionEndpointKind = eqx.field(static=True)
    end_kind: IntersectionEndpointKind = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    _fully_certified: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        first: SurfaceRegion,
        second: SurfaceRegion,
        *,
        chart_axes: ArrayLike,
        chart_start: ArrayLike,
        chart_end: ArrayLike,
        chart_lower: ArrayLike,
        chart_upper: ArrayLike,
        node_axes: ArrayLike,
        node_values: ArrayLike,
        node_lower: ArrayLike,
        node_upper: ArrayLike,
        node_period_shifts: ArrayLike,
        closed: bool,
        start_kind: IntersectionEndpointKind,
        end_kind: IntersectionEndpointKind,
    ) -> None:
        axes = np.asarray(chart_axes)
        start = np.asarray(chart_start, dtype=np.float64)
        end = np.asarray(chart_end, dtype=np.float64)
        lower = np.asarray(chart_lower, dtype=np.float64)
        upper = np.asarray(chart_upper, dtype=np.float64)
        n = axes.size if axes.ndim == 1 else -1
        if not isinstance(first, SurfaceRegion) or not isinstance(second, SurfaceRegion):
            raise TypeError("IntersectionCurve requires two source SurfaceRegion values.")
        if (
            n < 1
            or not np.issubdtype(axes.dtype, np.integer)
            or np.any((axes < 0) | (axes > 3))
        ):
            raise ValueError(
                "IntersectionCurve.chart_axes must select one of four coupled parameters per chart."
            )
        for name, values in (
            ("chart_start", start),
            ("chart_end", end),
            ("chart_lower", lower),
            ("chart_upper", upper),
        ):
            if values.shape != (n, 4) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"IntersectionCurve.{name} must be finite with shape (charts, 4)."
                )
        if np.any(lower >= upper):
            raise ValueError(
                "IntersectionCurve chart boxes must have positive width in all parameters."
            )
        if (
            np.any(start < lower)
            or np.any(start > upper)
            or np.any(end < lower)
            or np.any(end > upper)
        ):
            raise ValueError(
                "IntersectionCurve chart endpoints must lie inside their certified boxes."
            )
        node_axis = np.asarray(node_axes)
        node_value = np.asarray(node_values, dtype=np.float64)
        node_lo = np.asarray(node_lower, dtype=np.float64)
        node_hi = np.asarray(node_upper, dtype=np.float64)
        node_shifts = np.asarray(node_period_shifts)
        nodes = n + 1
        if (
            node_axis.shape != (nodes,)
            or not np.issubdtype(node_axis.dtype, np.integer)
            or np.any((node_axis < 0) | (node_axis > 3))
            or node_value.shape != (nodes,)
            or node_lo.shape != (nodes, 4)
            or node_hi.shape != (nodes, 4)
            or not np.all(np.isfinite(node_value))
            or not np.all(np.isfinite(node_lo))
            or not np.all(np.isfinite(node_hi))
            or np.any(node_lo >= node_hi)
        ):
            raise ValueError(
                "IntersectionCurve node root witnesses must be finite aligned 4D boxes."
            )
        if node_shifts.shape != (nodes, 2, 4) or not np.issubdtype(
            node_shifts.dtype, np.integer
        ):
            raise ValueError(
                "IntersectionCurve.node_period_shifts must be integer incoming/outgoing shifts per node."
            )
        periods = np.asarray(
            tuple(
                0.0 if value is None else float(value)
                for value in (*first.patch.periods, *second.patch.periods)
            ),
            dtype=np.float64,
        )
        start_kind_ = parse(start_kind, IntersectionEndpointKind, "start_kind")
        end_kind_ = parse(end_kind, IntersectionEndpointKind, "end_kind")
        if not isinstance(closed, bool):
            raise TypeError("IntersectionCurve.closed must be bool.")
        if closed != (start_kind_ == "closed" and end_kind_ == "closed"):
            raise ValueError("Closed branches must mark both branch endpoints as closed.")
        pair = SurfacePairSystem(first.patch, second.patch)
        prepared = _prepare_pair(pair, source_regions=(first, second))
        family = orthogonal_cylinder_family(first, second)
        analytic_certificate = (
            None
            if family is None
            else _orthogonal_cylinder_curve_certificate(
                family,
                prepared,
                axes,
                start,
                end,
                lower,
                upper,
                node_axis,
                node_value,
                node_lo,
                node_hi,
                closed,
                start_kind_,
                end_kind_,
            )
        )
        if family is not None and analytic_certificate is None:
            raise ValueError(
                "An orthogonal-cylinder branch must match its closed-form source certificate."
            )
        if analytic_certificate is None:
            chart_preconditioners = np.empty((n, 3, 3), dtype=np.float64)
            chart_contractions = np.empty((n,), dtype=np.float64)
        else:
            chart_preconditioners, chart_contractions = analytic_certificate
        for index in range(n):
            for offset, region in ((0, first), (2, second)):
                region_lower = np.asarray(region.parameter_box[0])
                region_upper = np.asarray(region.parameter_box[1])
                for endpoint in (start[index], end[index]):
                    for local_axis, period in enumerate(region.patch.periods):
                        axis = offset + local_axis
                        value = float(endpoint[axis])
                        if region_lower[local_axis] <= value <= region_upper[local_axis]:
                            continue
                        if (
                            period is not None
                            and region.patch.native_period_symbols()[local_axis]
                            == "two_pi"
                            and Fraction.from_float(float(region_upper[local_axis]))
                            - Fraction.from_float(float(region_lower[local_axis]))
                            == Fraction.from_float(float(period))
                        ):
                            # A complete exact native turn admits every source phase
                            # without moving or clamping its authored endpoint.
                            continue
                        raise ValueError(
                            "IntersectionCurve chart endpoint lies outside its source region and period deck."
                        )
        if analytic_certificate is None:
            for index in range(n):
                witness = krawczyk(
                    prepared,
                    lower[index : index + 1],
                    upper[index : index + 1],
                    parameter_axis=int(axes[index]),
                )
                if not bool(witness.certified[0]):
                    raise ValueError(
                        "IntersectionCurve chart does not certify a unique regular source graph."
                    )
                chart_preconditioners[index] = witness.preconditioner[0]
                chart_contractions[index] = witness.contraction[0]
        for index in range(nodes):
            if not (
                node_lo[index, node_axis[index]]
                <= node_value[index]
                <= node_hi[index, node_axis[index]]
            ):
                raise ValueError(
                    "IntersectionCurve node witness does not contain its fixed source coordinate."
                )
            if analytic_certificate is None:
                node_system = _NodeSystem(
                    pair, int(node_axis[index]), float(node_value[index])
                )
                node_prepared = (
                    _ExactAffineNodePrepared(
                        prepared,
                        int(node_axis[index]),
                        float(node_value[index]),
                    )
                    if isinstance(prepared, _ExactAffinePairPrepared)
                    else _PreparedNode(
                        node_system,
                        prepared,
                        source_regions=(first, second),
                    )
                )
                witness = krawczyk(
                    node_prepared,
                    node_lo[index : index + 1],
                    node_hi[index : index + 1],
                )
                if not bool(witness.certified[0]):
                    raise ValueError(
                        "IntersectionCurve chart endpoint is not an isolated unique source root."
                    )
            if index < n:
                outgoing = node_shifts[index, 1] * periods
                if np.any(start[index] < node_lo[index] + outgoing) or np.any(
                    start[index] > node_hi[index] + outgoing
                ):
                    raise ValueError(
                        "IntersectionCurve chart start must bind its isolated node through the declared periodic deck."
                    )
            if index > 0:
                incoming = node_shifts[index, 0] * periods
                if np.any(end[index - 1] < node_lo[index] + incoming) or np.any(
                    end[index - 1] > node_hi[index] + incoming
                ):
                    raise ValueError(
                        "IntersectionCurve chart end must bind its isolated node through the declared periodic deck."
                    )
        self.first, self.second = first, second
        self.chart_axes = jnp.asarray(axes, dtype=jnp.int32)
        self.chart_start = jnp.asarray(start, dtype=jnp.float64)
        self.chart_end = jnp.asarray(end, dtype=jnp.float64)
        self.chart_lower = jnp.asarray(lower, dtype=jnp.float64)
        self.chart_upper = jnp.asarray(upper, dtype=jnp.float64)
        self.preconditioners = jnp.asarray(chart_preconditioners, dtype=jnp.float64)
        self.contractions = jnp.asarray(chart_contractions, dtype=jnp.float64)
        self.node_axes = jnp.asarray(node_axis, dtype=jnp.int32)
        self.node_values = jnp.asarray(node_value, dtype=jnp.float64)
        self.node_lower = jnp.asarray(node_lo, dtype=jnp.float64)
        self.node_upper = jnp.asarray(node_hi, dtype=jnp.float64)
        self.node_period_shifts = jnp.asarray(node_shifts, dtype=jnp.int32)
        self.closed = closed
        self.start_kind = start_kind_
        self.end_kind = end_kind_
        self._fully_certified = analytic_certificate is not None or bool(
            np.all(chart_contractions < 1.0)
        )
        self.branch_id = canonical_fingerprint(
            {
                "kind": "source-defined-intersection-branch",
                "first": first.identity_payload(),
                "second": second.identity_payload(),
                "chart_axes": array_tree_fingerprint(self.chart_axes),
                "chart_start": array_tree_fingerprint(self.chart_start),
                "chart_end": array_tree_fingerprint(self.chart_end),
                "chart_lower": array_tree_fingerprint(self.chart_lower),
                "chart_upper": array_tree_fingerprint(self.chart_upper),
                "node_axes": array_tree_fingerprint(self.node_axes),
                "node_values": array_tree_fingerprint(self.node_values),
                "node_lower": array_tree_fingerprint(self.node_lower),
                "node_upper": array_tree_fingerprint(self.node_upper),
                "node_period_shifts": array_tree_fingerprint(self.node_period_shifts),
                "closed": closed,
                "endpoints": (start_kind_, end_kind_),
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def num_charts(self) -> int:
        return int(self.chart_axes.shape[0])

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return 0.0, float(self.num_charts)

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.parameter_interval

    @property
    def period(self) -> float | None:
        return float(self.num_charts) if self.closed else None

    def shift_parameter_bounds(
        self, shifts: ArrayLike, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enclose only declared integer two-pi source-period deck shifts."""
        values = np.asarray(shifts)
        if values.shape != (4,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(
                "IntersectionCurve shifts must contain four integer source gauges."
            )
        periods = (*self.first.patch.periods, *self.second.patch.periods)
        symbols = (
            *self.first.patch.native_period_symbols(),
            *self.second.patch.native_period_symbols(),
        )

        lower = np.zeros((4,), dtype=np.float64)
        upper = np.zeros((4,), dtype=np.float64)
        for axis, shift in enumerate(values.tolist()):
            if shift == 0:
                continue
            if periods[axis] is None or symbols[axis] != "two_pi":
                raise ValueError(
                    "IntersectionCurve deck shifts require the declared native two_pi period."
                )
            bounds = period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(int(shift))))
            lower[axis], upper[axis] = float(bounds[0]), float(bounds[1])
        return lower, upper

    def node_exact_coordinates(self, node: int, /) -> tuple[tuple[int, float], ...]:
        """Return the source coordinates that one node's root has exactly.

        Every node fixes one coordinate. A closed-form cylinder start node may
        also be proved to lie exactly on the large cylinder's seam, when its
        enclosure sits on the seam's own sheet.
        """
        exact = ((int(self.node_axes[node]), float(self.node_values[node])),)
        if node != 0 or not self.closed:
            return exact
        family = orthogonal_cylinder_family(self.first, self.second)
        if family is None:
            return exact
        branch_sign = _orthogonal_cylinder_curve_branch_sign(self, family)
        if branch_sign is None or not orthogonal_cylinder_exact_seam_start(
            family, branch_sign
        ):
            return exact
        angle_axis = 2 * family.large_index
        seam = float(family.source_regions[family.large_index].parameter_box[0, 0])
        if not (
            float(self.node_lower[0, angle_axis])
            <= seam
            <= float(self.node_upper[0, angle_axis])
        ):
            return exact
        return (*exact, (angle_axis, seam))

    def closure_period_shifts(self) -> np.ndarray:
        """Return the integer source deck from the start node to the closing node.

        A closed branch's first and last nodes enclose the same coupled root,
        possibly on different native-period sheets. Each periodic axis must
        admit exactly one integer turn under which the two enclosures meet;
        any other configuration is not a closed source branch.
        """
        shifts = np.zeros((4,), dtype=np.int64)
        if not self.closed:
            return shifts
        last = self.num_charts
        start_lower = np.asarray(self.node_lower[0], dtype=np.float64)
        start_upper = np.asarray(self.node_upper[0], dtype=np.float64)
        end_lower = np.asarray(self.node_lower[last], dtype=np.float64)
        end_upper = np.asarray(self.node_upper[last], dtype=np.float64)
        periods = (*self.first.patch.periods, *self.second.patch.periods)
        symbols = (
            *self.first.patch.native_period_symbols(),
            *self.second.patch.native_period_symbols(),
        )
        for axis in range(4):
            if (
                start_lower[axis] <= end_upper[axis]
                and end_lower[axis] <= start_upper[axis]
            ):
                continue
            if periods[axis] is None or symbols[axis] != "two_pi":
                raise ValueError(
                    "A closed IntersectionCurve must close on a declared native period."
                )
            estimate = round(
                (
                    0.5 * (end_lower[axis] + end_upper[axis])
                    - 0.5 * (start_lower[axis] + start_upper[axis])
                )
                / math.tau
            )
            turns = []
            for turn in range(estimate - 1, estimate + 2):
                lower, upper = self.shift_parameter_bounds(
                    np.asarray(
                        [turn if index == axis else 0 for index in range(4)],
                        dtype=np.int64,
                    )
                )
                shifted_lower = math.nextafter(
                    float(start_lower[axis] + lower[axis]), -math.inf
                )
                shifted_upper = math.nextafter(
                    float(start_upper[axis] + upper[axis]), math.inf
                )
                if shifted_lower <= end_upper[axis] and end_lower[axis] <= shifted_upper:
                    turns.append(turn)
            if len(turns) != 1:
                raise ValueError(
                    "A closed IntersectionCurve must close under one native period turn."
                )
            shifts[axis] = turns[0]
        return shifts

    def parameter_enclosures(self, first: float, last: float, /) -> np.ndarray:
        """Enclose all four source-chart parameters over a certified branch interval."""
        if first == last:
            if not math.isfinite(first) or not 0.0 <= first <= self.num_charts:
                raise ValueError(
                    "A branch parameter must lie inside its finite source interval."
                )
        else:
            self.validate_range(first, last)
        if not self.fully_certified:
            raise ValueError(
                "A branch parameter enclosure requires a fully certified source branch."
            )
        lower_parameter, upper_parameter = float(first), float(last)
        if upper_parameter == lower_parameter:
            first_chart = min(int(math.floor(lower_parameter)), self.num_charts - 1)
            last_chart = first_chart
        else:
            first_chart = min(int(math.floor(lower_parameter)), self.num_charts - 1)
            last_chart = min(
                int(math.floor(math.nextafter(upper_parameter, -math.inf))),
                self.num_charts - 1,
            )
        family = orthogonal_cylinder_family(self.first, self.second)
        if family is not None:
            branch_sign = _orthogonal_cylinder_curve_branch_sign(self, family)
            if branch_sign is None:
                raise ValueError("The exact cylinder branch has no isolated source sign.")
            theta_axis = 2 * family.small_index
            large_angle_axis = 2 * family.large_index
            boxes: list[np.ndarray] = []
            for chart in range(first_chart, last_chart + 1):
                chart_first = max(lower_parameter, float(chart))
                chart_last = min(upper_parameter, float(chart + 1))
                start = np.asarray(self.chart_start[chart], dtype=np.float64)
                end = np.asarray(self.chart_end[chart], dtype=np.float64)
                start_axis = Fraction.from_float(float(start[theta_axis]))
                end_axis = Fraction.from_float(float(end[theta_axis]))
                first_fraction = Fraction.from_float(chart_first - chart)
                last_fraction = Fraction.from_float(chart_last - chart)
                exact_first = start_axis + first_fraction * (end_axis - start_axis)
                exact_last = start_axis + last_fraction * (end_axis - start_axis)
                exact_lower, exact_upper = (
                    min(exact_first, exact_last),
                    max(exact_first, exact_last),
                )
                theta_lower, theta_upper = (
                    float(exact_lower),
                    float(exact_upper),
                )
                if Fraction.from_float(theta_lower) > exact_lower:
                    theta_lower = math.nextafter(theta_lower, -math.inf)
                if Fraction.from_float(theta_upper) < exact_upper:
                    theta_upper = math.nextafter(theta_upper, math.inf)
                # A seam node's root lies at an irrational small angle that its
                # chart endpoint only represents; a chart end enclosure also
                # encloses that node's whole small-angle range.
                for node, touches in (
                    (chart, chart_first == chart),
                    (chart + 1, chart_last == chart + 1),
                ):
                    if touches and int(self.node_axes[node]) == large_angle_axis:
                        theta_lower = min(
                            theta_lower, float(self.node_lower[node, theta_axis])
                        )
                        theta_upper = max(
                            theta_upper, float(self.node_upper[node, theta_axis])
                        )
                bounds = orthogonal_cylinder_graph_bounds(
                    family,
                    theta_lower,
                    theta_upper,
                    branch_sign,
                    float(start[large_angle_axis]),
                )
                if bounds is None:
                    raise ValueError(
                        "The exact cylinder branch interval has no outward source enclosure."
                    )
                boxes.append(np.stack(bounds))
            if not boxes:
                raise ValueError(
                    "A branch parameter interval must intersect at least one source chart."
                )
            return np.stack(boxes)
        prepared = _branch_prepared_pair(self)
        boxes: list[np.ndarray] = []
        for chart in range(first_chart, last_chart + 1):
            chart_first = max(lower_parameter, float(chart))
            chart_last = min(upper_parameter, float(chart + 1))
            axis = int(self.chart_axes[chart])
            free_axes = tuple(index for index in range(4) if index != axis)
            start = np.asarray(self.chart_start[chart], dtype=np.float64)
            end = np.asarray(self.chart_end[chart], dtype=np.float64)
            start_axis = Fraction.from_float(float(start[axis]))
            end_axis = Fraction.from_float(float(end[axis]))
            first_fraction = Fraction.from_float(chart_first - chart)
            last_fraction = Fraction.from_float(chart_last - chart)
            exact_first = start_axis + first_fraction * (end_axis - start_axis)
            exact_last = start_axis + last_fraction * (end_axis - start_axis)
            exact_lower, exact_upper = (
                min(exact_first, exact_last),
                max(exact_first, exact_last),
            )
            axis_lower, axis_upper = float(exact_lower), float(exact_upper)
            if Fraction.from_float(axis_lower) > exact_lower:
                axis_lower = math.nextafter(axis_lower, -math.inf)
            if Fraction.from_float(axis_upper) < exact_upper:
                axis_upper = math.nextafter(axis_upper, math.inf)
            lower_box = np.asarray(self.chart_lower[chart], dtype=np.float64).copy()
            upper_box = np.asarray(self.chart_upper[chart], dtype=np.float64).copy()
            lower_box[axis], upper_box[axis] = axis_lower, axis_upper
            witness = krawczyk(
                prepared,
                lower_box[None, :],
                upper_box[None, :],
                parameter_axis=axis,
            )
            if not bool(witness.certified[0]):
                raise ValueError(
                    "The branch subinterval does not certify a unique source graph."
                )
            contracted_lower = lower_box.copy()
            contracted_upper = upper_box.copy()
            contracted_lower[list(free_axes)] = witness.lower[0]
            contracted_upper[list(free_axes)] = witness.upper[0]
            if isinstance(prepared, _ExactAffinePairPrepared):
                prepared.intersect_exact_coordinates(contracted_lower, contracted_upper)
            # A node fixing another coordinate (the trim edge a boundary branch
            # ends on) is only represented by its chart endpoint; an enclosure
            # reaching that node also encloses the node's own root.
            for node, touches in (
                (chart, chart_first == chart),
                (chart + 1, chart_last == chart + 1),
            ):
                if touches and int(self.node_axes[node]) != axis:
                    contracted_lower = np.minimum(
                        contracted_lower, np.asarray(self.node_lower[node])
                    )
                    contracted_upper = np.maximum(
                        contracted_upper, np.asarray(self.node_upper[node])
                    )
            boxes.append(np.stack((contracted_lower, contracted_upper)))
        if not boxes:
            raise ValueError(
                "A branch parameter interval must intersect at least one source chart."
            )
        return np.stack(boxes)

    @property
    def fully_certified(self) -> bool:
        return self._fully_certified

    def is_c1_on(self, first: float, last: float, /) -> bool:
        """Certify C1 only on one regular implicit graph chart.

        Chart seams retain root continuity, but without a derivative-match
        witness an interval crossing a seam is not advertised as C1.
        """
        if not (
            math.isfinite(first)
            and math.isfinite(last)
            and 0.0 <= first <= last <= self.num_charts
        ):
            raise ValueError(
                "IntersectionCurve range must be nondecreasing inside its branch interval."
            )
        if first == last or not self.fully_certified:
            return False
        if first == 0.0 and self.start_kind in ("singular", "uncertified"):
            return False
        if last == self.num_charts and self.end_kind in ("singular", "uncertified"):
            return False
        if self.closed and first == 0.0 and last == self.num_charts:
            return False
        first_chart = min(int(math.floor(first)), self.num_charts - 1)
        last_chart = min(
            int(math.floor(math.nextafter(last, -math.inf))), self.num_charts - 1
        )
        if first_chart != last_chart:
            return False
        try:
            parameter_boxes = self.parameter_enclosures(first, last)
            return all(
                self.first.patch.is_c1_on(np.stack((box[0, :2], box[1, :2])))
                and self.second.patch.is_c1_on(np.stack((box[0, 2:], box[1, 2:])))
                for box in parameter_boxes
            )
        except (
            ArithmeticError,
            np.linalg.LinAlgError,
            TypeError,
            ValueError,
            NotImplementedError,
        ):
            return False

    def evaluate_coupled(self, parameters: ArrayLike, /) -> IntersectionCurvePoint:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim > 0 and values.shape[-1:] == (1,):
            values = values[..., 0]
        if not isinstance(values, Tracer):
            host = np.asarray(values, dtype=np.float64)
            if (
                np.any(~np.isfinite(host))
                or np.any(host < 0.0)
                or np.any(host > self.num_charts)
            ):
                raise ValueError(
                    "IntersectionCurve parameters must lie inside its finite branch interval."
                )
        scalar = values.ndim == 0
        flat = jnp.reshape(values, (-1,))
        family = orthogonal_cylinder_family(self.first, self.second)
        if family is None:
            result = jax.vmap(_evaluate_curve_point, in_axes=(None, 0))(self, flat)
        elif family.large_index == 0:
            result = jax.vmap(
                _evaluate_orthogonal_cylinder_curve_point_large_first,
                in_axes=(None, 0),
            )(self, flat)
        else:
            result = jax.vmap(
                _evaluate_orthogonal_cylinder_curve_point_large_second,
                in_axes=(None, 0),
            )(self, flat)
        shape = values.shape
        restored = tuple(value.reshape((*shape, *value.shape[1:])) for value in result)
        if scalar:
            restored = tuple(
                value.reshape(value.shape[len(shape) :]) for value in restored
            )
        return IntersectionCurvePoint(*restored)

    def evaluate(self, parameters: Array, /) -> Array:
        return self.evaluate_coupled(parameters).point

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        """Enclose source points from certified branch chart/node boxes."""
        if (
            not math.isfinite(first)
            or not math.isfinite(last)
            or first < 0.0
            or first > last
            or last > self.num_charts
        ):
            raise ValueError(
                "IntersectionCurve bounding range must be ordered inside its branch interval."
            )
        if not self.fully_certified:
            raise ValueError(
                "IntersectionCurve bounding boxes require a fully certified source branch."
            )

        if first == last and first == round(first):
            node = int(first)
            parameter_boxes = (
                np.stack(
                    (
                        np.asarray(self.node_lower[node], dtype=np.float64),
                        np.asarray(self.node_upper[node], dtype=np.float64),
                    )
                ),
            )
        else:
            first_chart = min(int(math.floor(first)), self.num_charts - 1)
            last_chart = (
                first_chart
                if first == last
                else min(
                    int(math.floor(math.nextafter(last, -math.inf))),
                    self.num_charts - 1,
                )
            )
            parameter_boxes = tuple(
                np.stack(
                    (
                        np.asarray(self.chart_lower[chart], dtype=np.float64),
                        np.asarray(self.chart_upper[chart], dtype=np.float64),
                    )
                )
                for chart in range(first_chart, last_chart + 1)
            )

        lower = np.full((3,), np.inf, dtype=np.float64)
        upper = np.full((3,), -np.inf, dtype=np.float64)
        for parameters in parameter_boxes:
            source_box = np.stack((parameters[0, :2], parameters[1, :2]))
            physical_box = np.asarray(
                self.first.patch.bounding_box(source_box), dtype=np.float64
            )
            if (
                physical_box.shape != (2, 3)
                or not np.all(np.isfinite(physical_box))
                or np.any(physical_box[0] > physical_box[1])
            ):
                raise ValueError(
                    "IntersectionCurve source patch returned an invalid physical enclosure."
                )
            lower = np.minimum(lower, physical_box[0])
            upper = np.maximum(upper, physical_box[1])
        return np.stack((lower, upper))

    def p_curve(self, side: IntersectionSide, /) -> IntersectionPCurve:
        side_ = parse(side, IntersectionSide, "side")
        return IntersectionPCurve(self, side_)

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        if not (
            math.isfinite(first)
            and math.isfinite(last)
            and 0.0 <= first < last <= self.num_charts
        ):
            raise ValueError(
                "IntersectionCurve range must be increasing inside its branch interval."
            )
        return float(first), float(last)

    def identity_payload(self) -> dict[str, object]:
        return {"kind": "intersection-curve", "branch_id": self.branch_id}


class IntersectionPCurve(AbstractCurve):
    """Source p-curve coupled to the exact same implicit root as its 3D branch."""

    curve: IntersectionCurve
    side: IntersectionSide = eqx.field(static=True)
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: IntersectionCurve,
        side: IntersectionSide,
        *,
        first: float | None = None,
        last: float | None = None,
        reversed: bool = False,
    ) -> None:
        side_ = parse(side, IntersectionSide, "side")
        first_ = 0.0 if first is None else float(first)
        last_ = float(curve.num_charts) if last is None else float(last)
        curve.validate_range(first_, last_)
        if not isinstance(reversed, bool):
            raise TypeError("IntersectionPCurve.reversed must be bool.")
        self.curve = curve
        self.side = side_
        self.first = first_
        self.last = last_
        self.reversed = reversed

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return self.first, self.last

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.parameter_interval

    @property
    def period(self) -> float | None:
        if self.curve.closed and self.first == 0.0 and self.last == self.curve.num_charts:
            return self.last - self.first
        return None

    def _chart_evaluator(self, chart: int, /) -> Callable[[Array], Array]:
        family = orthogonal_cylinder_family(self.curve.first, self.curve.second)
        if family is not None:
            branch_sign = _orthogonal_cylinder_curve_branch_sign(self.curve, family)
            if branch_sign is None:
                raise ValueError(
                    "An analytic cylinder p-curve requires a proved branch sign."
                )
            angle_data = _orthogonal_cylinder_chart_angle_data(
                self.curve, family, chart, branch_sign
            )
            if angle_data is not None:
                angle_mode, angle_sign, angle_turns = angle_data

                def evaluate(parameter: Array, /) -> Array:
                    carrier = (
                        self.first + self.last - parameter if self.reversed else parameter
                    )
                    parameters = _evaluate_orthogonal_cylinder_parameters_on_chart(
                        self.curve,
                        chart,
                        carrier,
                        family.large_index,
                        branch_sign,
                        angle_mode,
                        angle_sign,
                        angle_turns,
                    )
                    return parameters[:2] if self.side == "first" else parameters[2:]

                return evaluate
        axis = int(np.asarray(self.curve.chart_axes, dtype=np.int64)[chart])

        def evaluate(parameter: Array, /) -> Array:
            carrier = self.first + self.last - parameter if self.reversed else parameter
            result = _evaluate_curve_point_on_fixed_chart(
                self.curve, chart, axis, carrier
            )
            return result[1] if self.side == "first" else result[2]

        return evaluate

    def _evaluate_on_chart(self, parameters: Array, chart: int, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim > 0 and values.shape[-1:] == (1,):
            values = values[..., 0]
        scalar = values.ndim == 0
        flat = jnp.reshape(values, (-1,))
        result = jax.vmap(self._chart_evaluator(chart))(flat)
        shape = values.shape
        restored = result.reshape((*shape, *result.shape[1:]))
        if scalar:
            restored = restored.reshape(restored.shape[len(shape) :])
        return restored

    def evaluate(self, parameters: Array, /) -> Array:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if values.ndim > 0 and values.shape[-1:] == (1,):
            values = values[..., 0]
        chart = int(self.first)
        if self.last <= chart + 1.0:
            return self._evaluate_on_chart(values, chart)
        carrier = self.first + self.last - values if self.reversed else values
        coupled = self.curve.evaluate_coupled(carrier)
        result = (
            coupled.first_parameters
            if self.side == "first"
            else coupled.second_parameters
        )
        last_chart = math.ceil(self.last) - 1
        if self.last == last_chart + 1 and self.last < self.curve.num_charts:
            # The span's terminal node lies on its own last chart's source
            # sheet, which a periodic transition may separate from the next's.
            terminal = self._evaluate_on_chart(values, last_chart)
            result = jnp.where((carrier >= self.last)[..., None], terminal, result)
        return result

    def is_c1_on(self, first: float, last: float, /) -> bool:
        AbstractCurve.is_c1_on(self, first, last)
        if self.reversed:
            branch_first = self.first + self.last - last
            branch_last = self.first + self.last - first
        else:
            branch_first, branch_last = first, last
        return self.curve.is_c1_on(branch_first, branch_last)

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        AbstractCurve.is_c1_on(self, first, last)
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Curve derivative bounds support order=1 only.")
        if not (
            math.isfinite(first)
            and math.isfinite(last)
            and self.first <= first <= last <= self.last
        ):
            raise ValueError(
                "IntersectionPCurve derivative bounds must lie inside its source interval."
            )
        branch_first, branch_last = (
            (self.first + self.last - last, self.first + self.last - first)
            if self.reversed
            else (first, last)
        )
        first_chart = min(int(math.floor(branch_first)), self.curve.num_charts - 1)
        last_chart = min(
            int(math.floor(math.nextafter(branch_last, -math.inf))),
            self.curve.num_charts - 1,
        )
        if branch_first == branch_last:
            last_chart = first_chart
        family = orthogonal_cylinder_family(self.curve.first, self.curve.second)
        branch_sign = (
            None
            if family is None
            else _orthogonal_cylinder_curve_branch_sign(self.curve, family)
        )
        if family is not None and branch_sign is None:
            raise ValueError(
                "IntersectionPCurve derivative bounds require a proved cylinder branch sign."
            )
        from .._rational import fraction_down, fraction_up

        derivative_lower: np.ndarray | None = None
        derivative_upper: np.ndarray | None = None
        for chart in range(first_chart, last_chart + 1):
            chart_first = max(branch_first, float(chart))
            chart_last = min(branch_last, float(chart + 1))
            if chart_first > chart_last:
                continue
            if family is None:
                lower4, upper4 = _implicit_chart_derivative_bounds(
                    self.curve, chart, chart_first, chart_last
                )
                if self.reversed:
                    lower4, upper4 = -upper4, -lower4
                columns = slice(0, 2) if self.side == "first" else slice(2, 4)
                current_lower = lower4[columns]
                current_upper = upper4[columns]
            else:
                theta_axis = 2 * family.small_index
                chart_start = np.asarray(self.curve.chart_start[chart], dtype=np.float64)
                chart_end = np.asarray(self.curve.chart_end[chart], dtype=np.float64)
                theta_start = Fraction.from_float(float(chart_start[theta_axis]))
                theta_end = Fraction.from_float(float(chart_end[theta_axis]))
                exact_theta_first = theta_start + (
                    Fraction.from_float(chart_first) - chart
                ) * (theta_end - theta_start)
                exact_theta_last = theta_start + (
                    Fraction.from_float(chart_last) - chart
                ) * (theta_end - theta_start)
                theta_lower = fraction_down(min(exact_theta_first, exact_theta_last))
                theta_upper = fraction_up(max(exact_theta_first, exact_theta_last))
                assert branch_sign is not None
                bounds = orthogonal_cylinder_graph_derivative_bounds(
                    family,
                    theta_lower,
                    theta_upper,
                    branch_sign,
                )
                if bounds is None:
                    raise ValueError(
                        "IntersectionPCurve cylinder derivative interval is unresolved."
                    )
                scale = theta_end - theta_start
                if self.reversed:
                    scale = -scale
                scale_interval = (
                    np.asarray(fraction_down(scale), dtype=np.float64),
                    np.asarray(fraction_up(scale), dtype=np.float64),
                )
                from .._interval_enclosure import interval_multiply

                current_lower, current_upper = interval_multiply(bounds, scale_interval)
                current_lower = np.asarray(current_lower, dtype=np.float64)
                current_upper = np.asarray(current_upper, dtype=np.float64)
                columns = slice(0, 2) if self.side == "first" else slice(2, 4)
                current_lower = current_lower[columns]
                current_upper = current_upper[columns]
            derivative_lower = (
                current_lower
                if derivative_lower is None
                else np.minimum(derivative_lower, current_lower)
            )
            derivative_upper = (
                current_upper
                if derivative_upper is None
                else np.maximum(derivative_upper, current_upper)
            )
        if derivative_lower is None or derivative_upper is None:
            raise ValueError("IntersectionPCurve derivative range has no source chart.")
        return derivative_lower, derivative_upper

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        if not (
            math.isfinite(first)
            and math.isfinite(last)
            and self.first <= first <= last <= self.last
        ):
            raise ValueError(
                "IntersectionPCurve bounds must lie inside its source interval."
            )
        if self.reversed:
            branch_first = self.first + self.last - last
            branch_last = self.first + self.last - first
        else:
            branch_first, branch_last = first, last
        if (
            branch_first == branch_last == self.last
            and self.first < self.last
            and float(self.last).is_integer()
        ):
            # The span's terminal node lies on its own last chart's source
            # sheet; a point query there would select the next chart.
            branch_first = max(self.first, math.nextafter(branch_last, -math.inf))
        boxes = self.curve.parameter_enclosures(branch_first, branch_last)
        columns = slice(0, 2) if self.side == "first" else slice(2, 4)
        lower = np.min(boxes[:, 0, columns], axis=0)
        upper = np.max(boxes[:, 1, columns], axis=0)
        return np.stack((lower, upper))

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": "intersection-pcurve",
            "branch_id": self.curve.branch_id,
            "side": self.side,
            "first": float(self.first).hex(),
            "last": float(self.last).hex(),
            "reversed": self.reversed,
        }


def _same_carrier_source(first: object, second: object, /) -> bool:
    if first is second:
        return True
    if type(first) is not type(second):
        return False
    first_identity = getattr(first, "identity_payload", None)
    second_identity = getattr(second, "identity_payload", None)
    if callable(first_identity) and callable(second_identity):
        return canonical_fingerprint(first_identity()) == canonical_fingerprint(
            second_identity()
        )
    return array_tree_fingerprint(first) == array_tree_fingerprint(second)


def _endpoint_binds_carrier(endpoint: object, carrier: AbstractCurve, /) -> bool:
    from ._boolean_core import affine_transport_source
    from ._intersection import (
        BranchRootEndpoint,
        ConicParameterEndpoint,
        CurveSurfaceIntersectionRoot,
        IntersectionCurvePointRoot,
        NativePeriodEndpoint,
        TrimIntersectionRoot,
        TrimRootEndpoint,
    )

    # A proved coincident chart transport keeps the source p-curve's parameter
    # and roots unchanged; bind both sides against that source carrier.
    source = affine_transport_source(
        carrier.curve if isinstance(carrier, CurveTrimSegment) else carrier
    )
    if isinstance(source, IntersectionPCurve) and source.reversed:
        if isinstance(endpoint, BranchRootEndpoint) or (
            isinstance(endpoint, TrimRootEndpoint)
            and (
                isinstance(endpoint.root, IntersectionCurvePointRoot)
                or (
                    isinstance(endpoint.root, CurveSurfaceIntersectionRoot)
                    and isinstance(endpoint.root.curve, IntersectionCurve)
                )
            )
        ):
            # Coupled roots use the branch scalar frame. Reverse the trim
            # traversal rather than changing only the source p-curve's map.
            return False
    if isinstance(endpoint, ConicParameterEndpoint):
        return _same_carrier_source(affine_transport_source(endpoint.carrier), source)
    if isinstance(endpoint, NativePeriodEndpoint):
        return _same_carrier_source(affine_transport_source(endpoint.curve), source)
    if isinstance(endpoint, BranchRootEndpoint):
        if endpoint.source_pcurve is not None and _same_carrier_source(
            endpoint.source_pcurve, source
        ):
            return True
        if isinstance(source, IntersectionPCurve):
            # source_side identifies the spatial edge's lift, not the branch
            # p-curve queried here; both charts share its scalar branch root.
            if not _same_carrier_source(source.curve, endpoint.curve):
                return False
            try:
                endpoint.source_image(include_periodic_shifts=False)
                first, last = endpoint.parameter_enclosure()
            except (ArithmeticError, TypeError, ValueError, NotImplementedError):
                return False
            return source.first <= first <= last <= source.last
        if isinstance(source, IntersectionCurve):
            if source.branch_id != endpoint.curve.branch_id:
                return False
            try:
                endpoint.source_image(include_periodic_shifts=False)
                first, last = endpoint.parameter_enclosure()
            except (ArithmeticError, TypeError, ValueError, NotImplementedError):
                return False
            return 0.0 <= first <= last <= source.num_charts
    if isinstance(endpoint, TrimRootEndpoint):
        if endpoint.source_pcurve is not None and _same_carrier_source(
            endpoint.source_pcurve, source
        ):
            return True
        if isinstance(endpoint.root, TrimIntersectionRoot):
            if endpoint.root.endpoint_witness is not None:
                witness = endpoint.root.endpoint_witness
                expected = (
                    witness.trim_curve
                    if endpoint.operand == "first"
                    else witness.source_pcurve
                )
                source_pcurve = endpoint.source_pcurve or expected
                from ._boolean_core import pcurve_affine_correspondence

                try:
                    return bool(
                        pcurve_affine_correspondence(expected, source_pcurve) is not None
                        and pcurve_affine_correspondence(source_pcurve, source)
                        is not None
                    )
                except (TypeError, ValueError, NotImplementedError):
                    return False
            operand = (
                endpoint.root.first
                if endpoint.operand == "first"
                else endpoint.root.second
            )
            if (
                isinstance(operand, CurveTrimSegment)
                and not operand.reversed
                and not endpoint.affine_transforms
                and _same_carrier_source(operand.curve, source)
            ):
                # A window-local root binds its carrier through the window's
                # exact forward chart map, carried by the endpoint itself.
                return Fraction.from_float(
                    endpoint.affine_parameter_offset
                ) == Fraction.from_float(operand.first) and Fraction.from_float(
                    endpoint.affine_parameter_scale
                ) == Fraction.from_float(operand.last) - Fraction.from_float(
                    operand.first
                )
            return _same_carrier_source(
                endpoint.root.first, source
            ) or _same_carrier_source(endpoint.root.second, source)
        if isinstance(endpoint.root, IntersectionCurvePointRoot):
            return (
                isinstance(source, IntersectionPCurve)
                and source.curve.branch_id == endpoint.root.curve.branch_id
            )
        if isinstance(endpoint.root, CurveSurfaceIntersectionRoot):
            return _same_carrier_source(endpoint.root.curve, source)
    return False


def _is_root_endpoint(endpoint: object, /) -> TypeGuard[intersection_owner.RootEndpoint]:
    from ._intersection import (
        BranchRootEndpoint,
        ConicParameterEndpoint,
        NativePeriodEndpoint,
        TrimRootEndpoint,
    )

    return isinstance(
        endpoint,
        (
            TrimRootEndpoint,
            BranchRootEndpoint,
            NativePeriodEndpoint,
            ConicParameterEndpoint,
        ),
    )


class CurveTrimSegment(AbstractCurve):
    """A source curve restricted and oriented by an exact trim interval."""

    curve: AbstractCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    reversed: bool = eqx.field(static=True)
    first_root: intersection_owner.RootEndpoint | None
    last_root: intersection_owner.RootEndpoint | None

    @checked
    def __init__(
        self,
        curve: AbstractCurve,
        first: float,
        last: float,
        *,
        reversed: bool = False,
        first_root: intersection_owner.RootEndpoint | None = None,
        last_root: intersection_owner.RootEndpoint | None = None,
    ) -> None:
        if not isinstance(curve, AbstractCurve) or curve.ambient_dimension != 2:
            raise ValueError("CurveTrimSegment.curve must be a planar AbstractCurve.")
        lower, upper = float(first), float(last)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("CurveTrimSegment interval must be finite and increasing.")
        domain = curve.parameter_domain
        if domain is not None and (lower < domain[0] or upper > domain[1]):
            raise ValueError("CurveTrimSegment interval must lie in its source domain.")
        if not isinstance(reversed, bool):
            raise TypeError("CurveTrimSegment.reversed must be bool.")
        endpoint_enclosures: list[tuple[float, float] | None] = [None, None]
        for index, (endpoint, parameter) in enumerate(
            ((first_root, lower), (last_root, upper))
        ):
            if endpoint is None:
                continue
            if not _endpoint_binds_carrier(endpoint, curve):
                raise ValueError(
                    "CurveTrimSegment endpoint root must bind its source carrier definition."
                )
            if not _is_root_endpoint(endpoint):
                raise TypeError(
                    "CurveTrimSegment endpoint must be a source root endpoint."
                )
            enclosure = endpoint.parameter_enclosure()
            if not enclosure[0] <= parameter <= enclosure[1]:
                raise ValueError(
                    "CurveTrimSegment endpoint representative must lie in its source root enclosure."
                )
            endpoint_enclosures[index] = enclosure
        if (
            endpoint_enclosures[0] is not None
            and endpoint_enclosures[1] is not None
            and endpoint_enclosures[0][1] >= endpoint_enclosures[1][0]
        ):
            raise ValueError(
                "CurveTrimSegment endpoint root enclosures must have proved order."
            )
        self.curve, self.first, self.last, self.reversed = curve, lower, upper, reversed
        self.first_root, self.last_root = first_root, last_root

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return 0.0, 1.0

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.parameter_interval

    def source_parameters(self, local: Array | float, /) -> Array | float:
        fraction = 1.0 - local if self.reversed else local
        return self.first + fraction * (self.last - self.first)

    def evaluate(self, parameters: Array, /) -> Array:
        return self.curve.evaluate(
            jnp.asarray(self.source_parameters(parameters), dtype=jnp.float64)
        )

    def is_c1_on(self, first: float, last: float, /) -> bool:
        AbstractCurve.is_c1_on(self, first, last)
        a, b = float(self.source_parameters(first)), float(self.source_parameters(last))
        return self.curve.is_c1_on(min(a, b), max(a, b))

    def derivative_bounds(
        self, first: float, last: float, /, *, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(order, bool) or not isinstance(order, int) or order != 1:
            raise ValueError("Curve derivative bounds support order=1 only.")
        if not (0.0 <= first <= last <= 1.0):
            raise ValueError("CurveTrimSegment query must lie inside its unit interval.")
        from .._interval_enclosure import interval_multiply
        from .._rational import fraction_down, fraction_up

        first_local, last_local = (
            Fraction.from_float(float(first)),
            Fraction.from_float(float(last)),
        )
        if self.reversed:
            first_local, last_local = 1 - last_local, 1 - first_local
        scale = Fraction.from_float(float(self.last - self.first))
        source_first = Fraction.from_float(float(self.first)) + first_local * scale
        source_last = Fraction.from_float(float(self.first)) + last_local * scale
        lower, upper = self.curve.derivative_bounds(
            fraction_down(min(source_first, source_last)),
            fraction_up(max(source_first, source_last)),
        )
        if self.reversed:
            scale = -scale
        lower_result, upper_result = interval_multiply(
            (lower, upper),
            (
                np.asarray(fraction_down(scale), dtype=np.float64),
                np.asarray(fraction_up(scale), dtype=np.float64),
            ),
        )
        return (
            np.asarray(lower_result, dtype=np.float64),
            np.asarray(upper_result, dtype=np.float64),
        )

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        if not (0.0 <= first <= last <= 1.0):
            raise ValueError("CurveTrimSegment query must lie inside its unit interval.")
        a, b = float(self.source_parameters(first)), float(self.source_parameters(last))
        lower, upper = min(a, b), max(a, b)
        box = np.asarray(self.curve.bounding_box(lower, upper), dtype=np.float64)
        # A trim end is a root whose representative parameter only lies in its
        # enclosure; a box reaching that end also encloses the root itself.
        domain = self.curve.parameter_domain
        for value, edge, root in (
            (lower, self.first, self.first_root),
            (upper, self.last, self.last_root),
        ):
            if value != edge or root is None:
                continue
            root_lower, root_upper = root.parameter_enclosure()
            if domain is not None:
                root_lower = max(root_lower, float(domain[0]))
                root_upper = min(root_upper, float(domain[1]))
            if root_lower > root_upper:
                continue
            root_box = np.asarray(
                self.curve.bounding_box(root_lower, root_upper), dtype=np.float64
            )
            box = np.stack(
                (np.minimum(box[0], root_box[0]), np.maximum(box[1], root_box[1]))
            )
        return box

    def identity_payload(self) -> dict[str, object]:
        from ._carrier_pieces import carrier_payload
        from ._intersection import endpoint_root_payload

        first_root, last_root = self.first_root, self.last_root
        if first_root is not None and not _is_root_endpoint(first_root):
            raise TypeError("CurveTrimSegment first root must be a source root endpoint.")
        if last_root is not None and not _is_root_endpoint(last_root):
            raise TypeError("CurveTrimSegment last root must be a source root endpoint.")
        return {
            "kind": "curve-trim-segment",
            "curve": carrier_payload(self.curve),
            "first": float(self.first).hex(),
            "last": float(self.last).hex(),
            "reversed": self.reversed,
            "first_root": None
            if first_root is None
            else endpoint_root_payload(first_root),
            "last_root": None if last_root is None else endpoint_root_payload(last_root),
        }


__all__ = [
    "CurveTrimSegment",
    "IntersectionCurve",
    "IntersectionCurvePoint",
    "IntersectionEndpointKind",
    "IntersectionPCurve",
    "IntersectionSide",
    "SurfacePairSystem",
    "SurfaceRegion",
]


def _exact_bspline_curve_value(
    curve: object, parameter: Fraction, /
) -> tuple[ExactScalar, ...] | None:
    from ._patches import BSplineCurve

    if not isinstance(curve, BSplineCurve):
        return None
    knots = curve.exact_knots
    weights = curve.exact_weights
    controls = curve.exact_control_points
    degree = curve.degree
    count = len(controls)
    last_parameter = knots[count]
    if parameter < knots[degree] or parameter > last_parameter:
        return None
    if parameter == last_parameter:
        span = count - 1
    else:
        span = next(
            (
                index
                for index in range(degree, count)
                if knots[index] <= parameter < knots[index + 1]
            ),
            -1,
        )
        if span < 0:
            return None
    homogeneous: list[tuple[ExactScalar, ...]] = [
        tuple(weight * value for value in control) + (weight,)
        for control, weight in zip(controls, weights, strict=True)
    ]
    values: list[tuple[ExactScalar, ...]] = [
        homogeneous[span - degree + index] for index in range(degree + 1)
    ]
    for order in range(1, degree + 1):
        for index in range(degree, order - 1, -1):
            knot_index = span - degree + index
            denominator = knots[knot_index + degree - order + 1] - knots[knot_index]
            alpha = (
                Fraction(0)
                if denominator == 0
                else (parameter - knots[knot_index]) / denominator
            )
            values[index] = tuple(
                (1 - alpha) * left + alpha * right
                for left, right in zip(values[index - 1], values[index], strict=True)
            )
    weight = values[degree][-1]
    if weight == 0:
        return None
    return tuple(value / weight for value in values[degree][:-1])


def exact_curve_parameter_point(
    curve: AbstractCurve,
    exact_parameter: Fraction | PeriodOffset,
    /,
) -> tuple[ExactScalar, ...] | None:
    """Evaluate supported source expressions in the exact real coefficient field."""
    from ...discretization._exact_algebraic_geometry import rational_coefficient
    from ._patches import (
        BSplineCurve,
        CircleCurve,
        EllipseCurve,
    )
    from ._placed import PlacedCurve
    from ._source_correspondence import _affine_curve_coefficients
    from ._trim_helpers import period_offset

    if isinstance(exact_parameter, PeriodOffset):
        rational, turns = exact_parameter.rational, exact_parameter.turns
    elif isinstance(exact_parameter, Fraction):
        rational, turns = exact_parameter, Fraction(0)
    else:
        return None
    if isinstance(curve, CurveTrimSegment):
        local = exact_parameter
        if curve.reversed:
            local = Fraction(1) - local
        endpoint = (
            curve.first_root if local == 0 else curve.last_root if local == 1 else None
        )
        from ._intersection import NativePeriodEndpoint

        if isinstance(endpoint, NativePeriodEndpoint):
            source_parameter = endpoint.exact_parameter
        else:
            first = Fraction.from_float(curve.first)
            delta = Fraction.from_float(curve.last) - first
            source_parameter = (
                period_offset(first + delta * local.rational, delta * local.turns)
                if isinstance(local, PeriodOffset)
                else first + delta * local
            )
        return exact_curve_parameter_point(curve.curve, source_parameter)
    if isinstance(curve, PlacedCurve):
        source = exact_curve_parameter_point(curve.definition, exact_parameter)
        if source is None or len(source) != 3:
            return None
        return curve.affine.apply((source[0], source[1], source[2]))
    if isinstance(curve, BSplineCurve) and not turns:
        return _exact_bspline_curve_value(curve, rational)
    if not turns:
        affine = _affine_curve_coefficients(
            curve, (Fraction(0),) * curve.ambient_dimension
        )
        if affine is not None:
            origin, direction = affine
            result: list[ExactScalar] = []
            for first, slope in zip(origin, direction, strict=True):
                value = first + rational * slope
                try:
                    result.append(rational_coefficient(value, role="curve source image"))
                except ValueError:
                    result.append(value)
            return tuple(result)
    quarter = turns * 4
    if rational != 0 or quarter.denominator != 1:
        return None
    phase = int(quarter) % 4
    cosine, sine = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(-1)),
    )[phase]
    if isinstance(curve, CircleCurve):
        center = curve.center_exact
        first = curve.first_axis_exact
        second = curve.second_axis_exact
        radius = curve.radius_exact
        return tuple(
            center[i] + radius * cosine * first[i] + radius * sine * second[i]
            for i in range(curve.ambient_dimension)
        )
    if isinstance(curve, EllipseCurve):
        center = curve.center_exact
        first = curve.first_axis_exact
        second = curve.second_axis_exact
        first_radius = curve.first_radius_exact
        second_radius = curve.second_radius_exact
        return tuple(
            center[i]
            + first_radius * cosine * first[i]
            + second_radius * sine * second[i]
            for i in range(curve.ambient_dimension)
        )
    return None


# The root owner imports these carriers. Load its module after their definitions;
# checked constructors resolve the qualified root union only on invocation.
from . import _intersection as intersection_owner


__all__ = [*__all__, "exact_curve_parameter_point"]
