#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host source evaluation and continuous derivative, chord, and radial-shell bounds."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
import phydrax.geometry._rational as rational_owner
import phydrax.geometry._source_interpolation as source_interpolation_owner
import phydrax.geometry.brep._constructors as constructors_owner
import phydrax.geometry.brep._exact as brep_exact_owner

from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..linalg import HermitianSpectrum
from ..typing import checked
from ._atlas import AbstractTrimCurve
from ._interval_enclosure import interval_add, interval_multiply, interval_subtract
from .brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    pcurve_periodic_source,
)
from .brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinRestrictionBank,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    meridian_frame,
    MeridianOverlaps,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    sphere_source_equivalence,
    SpherePatch,
    surface_differential,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from .brep._placed import PlacedCurve, PlacedSurface, source_transform_bounds
from .brep._request_preparations import TrimRequestPreparations


_SOURCE_BATCH_FLOOR = 64


@contextmanager
def source_query_request(
    queries_upper: int,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> Iterator[CoordinateEnclosureBudget]:
    """Bind a numerical source query batch to its explicit row-count allowance."""
    if budget is not None:
        yield budget
        return
    ticket = HostBatchTicket(0, queries_upper=queries_upper)
    view = CoordinateEnclosureBudget(ticket, maximum_work_units=0, maximum_memory_bytes=0)
    with ticket, view.activate():
        yield view


def consume_source_queries(count: int, /, *, budget: CoordinateEnclosureBudget) -> None:
    """Count a source-evaluation batch locally before its numerical execution."""
    if count > budget.ticket.remaining_geometry_queries:
        budget.ticket.refuse(0, queries_upper=count)
    budget.ticket.consume(geometry_queries=count)


def source_batch_bucket(count: int, /) -> int:
    """Bounded padded capacity of a host batch of ``count`` source queries."""
    return max(_SOURCE_BATCH_FLOOR, 1 << max(count - 1, 0).bit_length())


def bucketed(values: np.ndarray, /) -> Array:
    """Nonempty query rows padded with copies of their first row to a bucket.

    Padding lanes repeat an admitted query and are sliced off before any
    host value is formed; they never reach a published result.
    """
    padded = np.empty(
        (source_batch_bucket(values.shape[0]), *values.shape[1:]), dtype=np.float64
    )
    padded[: values.shape[0]] = values
    padded[values.shape[0] :] = values[0]
    return jax.device_put(padded)


@eqx.filter_jit
def source_evaluation(surface: AbstractSurfacePatch, charts: Array, /) -> Array:
    """One owned source program, with scientific coefficients kept dynamic."""
    return surface.evaluate(charts)


@eqx.filter_jit
def source_differential(surface: AbstractSurfacePatch, charts: Array, /) -> Array:
    """One owned source-differential program, coefficients kept dynamic."""
    return surface_differential(surface, charts)


@eqx.filter_jit
def curve_evaluation(curve: AbstractCurve, parameters: Array, /) -> Array:
    """One owned source-curve program, coefficients kept dynamic."""
    return curve.evaluate(parameters)


@checked
def host_surface_evaluation(
    surface: AbstractSurfacePatch, charts: np.ndarray, /
) -> np.ndarray:
    values = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
    if not values.shape[0]:
        return np.empty((0, 3), dtype=np.float64)
    result = source_evaluation(surface, bucketed(values))
    host = jax.device_get(result)
    return np.asarray(host, dtype=np.float64)[: values.shape[0]]


@checked
def host_surface_differential(
    surface: AbstractSurfacePatch, charts: np.ndarray, /
) -> np.ndarray:
    values = np.asarray(charts, dtype=np.float64).reshape((-1, 2))
    if not values.shape[0]:
        return np.empty((0, 3, 2), dtype=np.float64)
    result = source_differential(surface, bucketed(values))
    host = jax.device_get(result)
    return np.asarray(host, dtype=np.float64)[: values.shape[0]]


def gram_spectrum_bounds(matrix: np.ndarray, /) -> tuple[float, float]:
    """Outward Gram eigenvalue enclosure with native spectral residual evidence."""
    gram = matrix.T @ matrix
    spectrum = HermitianSpectrum(jnp.asarray(gram, dtype=jnp.float64))
    valid, host_values, host_vectors = jax.device_get(
        (spectrum.valid, spectrum.eigenvalues, spectrum.eigenvectors)
    )
    if not bool(np.asarray(valid)):
        raise ValueError(
            "MeshingDomain.derivative_gram must have a finite Hermitian spectrum; got invalid spectral evidence"
        )
    values = np.asarray(host_values)
    vectors = np.asarray(host_vectors)
    eps = np.finfo(np.float64).eps
    orthogonal_error = np.max(
        np.sum(
            np.abs(vectors.T @ vectors - np.eye(values.size))
            + 32 * eps * (np.abs(vectors).T @ np.abs(vectors)),
            axis=1,
        )
    )
    reconstruction = (vectors * values[None]) @ vectors.T
    roundoff = (
        64
        * eps
        * (
            np.abs(matrix).T @ np.abs(matrix)
            + (np.abs(vectors) * np.abs(values)[None]) @ np.abs(vectors).T
        )
    )
    residual = np.max(np.sum(np.abs(gram - reconstruction) + roundoff, axis=1))
    lower = float(values[0] * (1 - orthogonal_error) - residual)
    upper = float(values[-1] * (1 + orthogonal_error) + residual)
    return float(np.nextafter(lower, -np.inf)), float(np.nextafter(upper, np.inf))


def matrix_norm_upper(matrix: np.ndarray, /) -> float:
    """Sharp outward operator norm from the native Gram spectral owner."""
    _, upper = gram_spectrum_bounds(matrix)
    return float(np.nextafter(np.sqrt(max(upper, 0.0)), np.inf))


def hessian_bounds(
    surface: AbstractSurfacePatch,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float, float] | None:
    """Return global norms of the three second parameter derivatives."""
    match surface:
        case PlacedSurface():
            bounds = hessian_bounds(surface.definition, budget=budget)
            factor = matrix_norm_upper(np.asarray(surface.rotation))
            return (
                None
                if bounds is None
                else (bounds[0] * factor, bounds[1] * factor, bounds[2] * factor)
            )
        case PlanePatch():
            return 0.0, 0.0, 0.0
        case _:
            pass
    sphere = sphere_source_equivalence(surface, budget=budget)
    if sphere is not None:
        original, exact_radius = sphere
        radius = float(np.nextafter(float(abs(exact_radius)), np.inf))
        radial = matrix_norm_upper(
            np.column_stack(
                (np.asarray(original.first_axis), np.asarray(original.second_axis))
            )
        )
        whole = matrix_norm_upper(
            np.column_stack(
                (
                    np.asarray(original.first_axis),
                    np.asarray(original.second_axis),
                    np.asarray(original.axis),
                )
            )
        )
        first = float(np.nextafter(radius * radial, np.inf))
        third = float(np.nextafter(radius * whole, np.inf))
        return first, first, third
    match surface:
        case OffsetSurface():
            # The lowered primitive is the same mathematical source map.
            equivalent = surface.analytic_equivalent(budget=budget)
            return (
                None if equivalent is None else hessian_bounds(equivalent, budget=budget)
            )
        case CylinderPatch() | TorusPatch():
            radial = matrix_norm_upper(
                np.column_stack(
                    (np.asarray(surface.first_axis), np.asarray(surface.second_axis))
                )
            )
            whole = matrix_norm_upper(
                np.column_stack(
                    (
                        np.asarray(surface.first_axis),
                        np.asarray(surface.second_axis),
                        np.asarray(surface.axis),
                    )
                )
            )
            match surface:
                case CylinderPatch():
                    return abs(float(surface.radius)) * radial, 0.0, 0.0
                case TorusPatch():
                    major, minor = (
                        abs(float(surface.major_radius)),
                        abs(float(surface.minor_radius)),
                    )
                    return (major + minor) * radial, minor * radial, minor * whole
        case _:
            return None


def curve_second_bound(
    curve: AbstractCurve | IntersectionCurve,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Return a global second-derivative norm or infinity for an unsupported carrier."""
    match curve:
        case PlacedCurve():
            return curve_second_bound(
                curve.definition, budget=budget
            ) * matrix_norm_upper(np.asarray(curve.rotation))
        case LineCurve():
            return 0.0
        case CircleCurve():
            return float(curve.radius) * matrix_norm_upper(
                np.column_stack(
                    (np.asarray(curve.first_axis), np.asarray(curve.second_axis))
                )
            )
        case SurfaceIsoparametricCurve():
            bounds = hessian_bounds(curve.surface, budget=budget)
            if bounds is not None:
                return bounds[2] if curve.fixed_axis == 0 else bounds[0]
            return np.inf
        case _:
            return np.inf


def affine_bspline_interval(
    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
) -> bool:
    """Prove one closed degree-one source span is affine."""
    match curve:
        case PlacedCurve():
            return affine_bspline_interval(curve.definition, first, last)
        case BSplineCurve() if curve.degree == 1:
            lower, upper = sorted((first, last))
            for piece in curve.bezier_pieces():
                ((start, end),) = piece.parameter_bounds
                if not start <= lower <= upper <= end:
                    continue
                controls = np.asarray(piece.homogeneous_controls)
                weights = controls[:, -1]
                return (
                    controls.shape[0] == 2
                    and weights[0] > 0.0
                    and weights[0] == weights[1]
                )
            return False
        case _:
            return False


def physical_curve_chord_bound(
    curve: AbstractCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
    *,
    known_second: float | None = None,
    budget: CoordinateEnclosureBudget,
) -> float:
    width = last - first
    if width == 0:
        return 0.0
    if affine_bspline_interval(curve, first, last):
        return 0.0
    if curve_c1_interval(curve, first, last, budget=budget):
        second = (
            curve_second_bound(curve, budget=budget)
            if known_second is None
            else known_second
        )
        if not np.isfinite(second):
            low, high = curve.derivative_bounds(first, last, order=2, budget=budget)
            second = float(np.linalg.norm(np.maximum(np.abs(low), np.abs(high))))
        return second * width * width / 8
    low, high = curve.derivative_bounds(first, last, order=1, budget=budget)
    return float(np.linalg.norm(np.nextafter(high - low, np.inf))) * width / 4


def chart_curve_chord_bound(
    surface: AbstractSurfacePatch,
    curve: AbstractCurve | AbstractTrimCurve,
    first: float,
    last: float,
    /,
    *,
    known_second: float | None = None,
    budget: CoordinateEnclosureBudget,
) -> float:
    width = last - first
    if width == 0:
        return 0.0
    if affine_bspline_interval(curve, first, last):
        hessian = hessian_bounds(surface, budget=budget)
        if hessian == (0.0, 0.0, 0.0):
            return 0.0
    if known_second is not None:
        return known_second * width * width / 8
    if isinstance(curve, LineCurve):
        hessian = hessian_bounds(surface, budget=budget)
        if hessian is not None:
            direction = np.abs(np.asarray(curve.direction))
            second = (
                hessian[0] * direction[0] ** 2
                + 2 * hessian[1] * direction[0] * direction[1]
                + hessian[2] * direction[1] ** 2
            )
            return second * width * width / 8
    if isinstance(surface, PlanePatch) and isinstance(curve, CircleCurve):
        axes = np.stack((np.asarray(surface.first_axis), np.asarray(surface.second_axis)))
        mapped = np.column_stack(
            (
                np.asarray(curve.first_axis) @ axes,
                np.asarray(curve.second_axis) @ axes,
            )
        )
        return float(curve.radius) * matrix_norm_upper(mapped) * width * width / 8
    box = curve.bounding_box(first, last, budget=budget)
    first_low, first_high = curve.derivative_bounds(first, last, order=1, budget=budget)
    surface_low, surface_high = surface.derivative_bounds(box, order=1, budget=budget)
    if not curve_c1_interval(curve, first, last, budget=budget):
        products = np.stack(
            (
                surface_low * first_low,
                surface_low * first_high,
                surface_high * first_low,
                surface_high * first_high,
            )
        )
        low = np.sum(np.min(products, axis=0), axis=1)
        high = np.sum(np.max(products, axis=0), axis=1)
        slack = (
            64
            * np.finfo(np.float64).eps
            * np.sum(np.max(np.abs(products), axis=0), axis=1)
        )
        diameter = np.nextafter(high - low + 2 * slack, np.inf)
        return float(np.linalg.norm(diameter)) * width / 4
    second_low, second_high = curve.derivative_bounds(first, last, order=2, budget=budget)
    hessian_low, hessian_high = surface.derivative_bounds(box, order=2, budget=budget)
    if not all(
        np.all(np.isfinite(bound))
        for bound in (
            first_low,
            first_high,
            second_low,
            second_high,
            surface_low,
            surface_high,
            hessian_low,
            hessian_high,
        )
    ):
        return np.inf
    direction = np.maximum(np.abs(first_low), np.abs(first_high))
    acceleration = np.maximum(np.abs(second_low), np.abs(second_high))
    speed = np.linalg.norm(np.maximum(np.abs(surface_low), np.abs(surface_high)), axis=0)
    hessian = np.linalg.norm(
        np.maximum(np.abs(hessian_low), np.abs(hessian_high)), axis=0
    )
    second = float(direction @ hessian @ direction + speed @ acceleration)
    return second * width * width / 8


def cross_interval(
    a_low: np.ndarray,
    a_high: np.ndarray,
    b_low: np.ndarray,
    b_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for first, second in ((1, 2), (2, 0), (0, 1)):
        positive = interval_multiply(
            (a_low[first], a_high[first]),
            (b_low[second], b_high[second]),
        )
        negative = interval_multiply(
            (a_low[second], a_high[second]),
            (b_low[first], b_high[first]),
        )
        bounds = interval_subtract(positive, negative)
        lower.append(float(bounds[0]))
        upper.append(float(bounds[1]))
    return np.asarray(lower), np.asarray(upper)


def normal_interval_diameter(lower: np.ndarray, upper: np.ndarray, /) -> float:
    """Gauss-set diameter of a whole unnormalized-normal interval box.

    A convex box separated from zero has normalization derivative norm at most
    ``1 / rho``. Its diameter ``D`` therefore bounds normalized chord length by
    ``D / rho``; the angular diameter is at most ``2 asin(D / (2 rho))``.
    squared distances preserve separation and ratio rounding even when
    jets have different scales. No second derivative or C1 hypothesis is needed.
    """
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return np.inf
    separated = np.maximum(np.maximum(lower, -upper), 0.0)
    distance_squared = sum(
        (Fraction(float(value)) ** 2 for value in separated),
        Fraction(),
    )
    if distance_squared == 0:
        return np.inf
    diameter_squared = sum(
        (
            (Fraction(float(high)) - Fraction(float(low))) ** 2
            for low, high in zip(lower, upper, strict=True)
        ),
        Fraction(),
    )
    ratio_squared = diameter_squared / (4 * distance_squared)
    if ratio_squared >= 1:
        return float(np.nextafter(np.pi, np.inf))

    radius_upper = brep_exact_owner.square_root_interval(ratio_squared)[1]
    angle = 2 * np.arcsin(radius_upper)
    return float(np.nextafter(angle * (1 + 32 * np.finfo(np.float64).eps), np.inf))


def jet_normal_turns(
    surface: AbstractSurfacePatch,
    boxes: np.ndarray,
    widths: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
    restriction_bank: BernsteinRestrictionBank,
    meridian_jets: list[MeridianOverlaps],
) -> np.ndarray:
    """Normal-cone or Taylor turn from the surface's whole interval jets."""
    result = []
    first_lower, first_upper = surface.derivative_bounds_batch(
        boxes,
        order=1,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )
    second_lower, second_upper = surface.derivative_bounds_batch(
        boxes,
        order=2,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )
    for low, high, second_low, second_high, width in zip(
        first_lower,
        first_upper,
        second_lower,
        second_upper,
        widths,
        strict=True,
    ):
        cross_low, cross_high = cross_interval(
            low[:, 0], high[:, 0], low[:, 1], high[:, 1]
        )
        cone = normal_interval_diameter(cross_low, cross_high)
        if not np.isfinite(cone):
            result.append(np.inf)
            continue
        separated = np.maximum(np.maximum(cross_low, -cross_high), 0.0)
        regularity = float(np.linalg.norm(separated))
        first = np.linalg.norm(np.maximum(np.abs(low), np.abs(high)), axis=0)
        second = np.linalg.norm(
            np.maximum(np.abs(second_low), np.abs(second_high)), axis=0
        )
        bound = cone
        if (
            np.isfinite(regularity)
            and regularity > 0
            and np.all(np.isfinite(first))
            and np.all(np.isfinite(second))
        ):
            speed = np.asarray(
                (
                    second[0, 0] * first[1] + first[0] * second[1, 0],
                    second[0, 1] * first[1] + first[0] * second[1, 1],
                )
            )
            taylor = float(speed @ width / regularity)
            if np.isfinite(taylor):
                bound = min(bound, taylor)
        result.append(bound)
    return np.asarray(result, dtype=np.float64)


def basis_distortion(
    surface: ConePatch | CylinderPatch | SpherePatch | TorusPatch, /
) -> float:
    return matrix_distortion(
        np.column_stack(
            (
                np.asarray(surface.first_axis),
                np.asarray(surface.second_axis),
                np.asarray(surface.axis),
            )
        )
    )


def matrix_distortion(matrix: np.ndarray, /) -> float:
    lower, upper = gram_spectrum_bounds(matrix)
    if lower <= 0:
        return np.inf
    return float(np.nextafter(np.sqrt(upper / lower), np.inf))


@final
class _SphereBoundaryFrame(StrictModule):
    """An analytic spherical source with its verified full-period pole/seam layout."""

    surface: AbstractSurfacePatch
    definition: SpherePatch
    radius: Fraction = eqx.field(static=True)


def _sphere_seams_valid(
    seams: list[domain_owner.PatchCurveUse],
    operation: AbstractSurfacePatch,
) -> bool:
    """Verify paired periodic seams represent the same transverse source line."""
    if all(isinstance(use.pcurve, LineCurve) for use in seams):
        return True
    factors = tuple(pcurve_periodic_source(use.pcurve, operation) for use in seams)
    match factors:
        case ((LineCurve() as a, first_turn), (LineCurve() as b, second_turn)):
            return not (
                not np.array_equal(np.asarray(a.origin), np.asarray(b.origin))
                or not np.array_equal(np.asarray(a.direction), np.asarray(b.direction))
                or float(a.direction[0]) != 0.0
                or first_turn[1]
                or second_turn[1]
                or abs(first_turn[0] - second_turn[0]) != 1
                or {seams[0].first, seams[0].last} != {seams[1].first, seams[1].last}
            )
        case _:
            return False


def _prepare_sphere_boundary_frame(
    domain: domain_owner.MeshingDomain,
    patch: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SphereBoundaryFrame | None:
    """Admit full spherical source coverage from its two poles and paired seam."""
    value = domain.patches[patch]
    surface = value.surface
    operation = surface.definition if isinstance(surface, PlacedSurface) else surface
    equivalence = sphere_source_equivalence(operation, budget=budget)
    if equivalence is None or len(value.loops) != 1 or len(value.loops[0]) != 4:
        return None
    definition, exact_radius = equivalence
    poles = [use for use in value.loops[0] if isinstance(use, domain_owner.PatchPoleUse)]
    seams = [use for use in value.loops[0] if isinstance(use, domain_owner.PatchCurveUse)]
    if len(poles) != 2 or len(seams) != 2 or seams[0].curve != seams[1].curve:
        return None
    if not _sphere_seams_valid(seams, operation):
        return None
    endpoints = np.concatenate(
        [domain_owner.use_endpoints(use) for use in value.loops[0]]
    )
    slack = 128 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(endpoints))))
    if (
        abs(float(np.ptp(endpoints[:, 0])) - 2 * np.pi) > slack
        or abs(float(np.min(endpoints[:, 1])) + np.pi / 2) > slack
        or abs(float(np.max(endpoints[:, 1])) - np.pi / 2) > slack
        or any(abs(use.start[1] - use.end[1]) > slack for use in poles)
    ):
        return None
    return _SphereBoundaryFrame(surface, definition, exact_radius)


def _sphere_world_frame(
    prepared: _SphereBoundaryFrame,
) -> tuple[np.ndarray, float, float] | None:
    """Enclose physical shell radii and placement-induced center roundoff."""
    surface, definition, exact_radius = (
        prepared.surface,
        prepared.definition,
        prepared.radius,
    )
    matrix = np.column_stack(
        (
            np.asarray(definition.first_axis),
            np.asarray(definition.second_axis),
            np.asarray(definition.axis),
        )
    )
    low, high = gram_spectrum_bounds(matrix)
    if low <= 0:
        return None
    represented = float(abs(exact_radius))
    radius_low = (
        float(np.nextafter(represented, -np.inf))
        if Fraction(represented) > abs(exact_radius)
        else represented
    )
    radius_high = (
        float(np.nextafter(represented, np.inf))
        if Fraction(represented) < abs(exact_radius)
        else represented
    )
    radius_low = float(
        np.nextafter(radius_low * np.nextafter(np.sqrt(low), -np.inf), -np.inf)
    )
    radius_high = float(
        np.nextafter(radius_high * np.nextafter(np.sqrt(high), np.inf), np.inf)
    )
    center = np.asarray(definition.center, dtype=np.float64)
    if isinstance(surface, PlacedSurface):
        rotation, translation = (
            np.asarray(surface.rotation),
            np.asarray(surface.translation),
        )
        pose_low, pose_high = gram_spectrum_bounds(rotation)
        if pose_low <= 0:
            return None
        center_low, center_high = source_transform_bounds(rotation, center, center)
        center = rotation @ center + translation
        center_low, center_high = interval_add(
            (center_low, center_high), (translation, translation)
        )
        center_error = float(
            np.linalg.norm(
                np.maximum(np.abs(center_low - center), np.abs(center_high - center))
            )
        )
        center_error = float(
            np.nextafter(center_error * (1 + 64 * np.finfo(np.float64).eps), np.inf)
        )
        radius_low = float(
            np.nextafter(
                radius_low * np.nextafter(np.sqrt(pose_low), -np.inf) - center_error,
                -np.inf,
            )
        )
        radius_high = float(
            np.nextafter(
                radius_high * np.nextafter(np.sqrt(pose_high), np.inf) + center_error,
                np.inf,
            )
        )
    if radius_low <= 0:
        return None
    return center, radius_low, radius_high


def full_sphere_frame(
    domain: domain_owner.MeshingDomain,
    patch: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, float, float] | None:
    """Return a radial shell frame after proving the domain patch spans a full sphere."""
    prepared = _prepare_sphere_boundary_frame(domain, patch, budget=budget)
    return None if prepared is None else _sphere_world_frame(prepared)


def interval_vector_norm(
    value: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    low, high = value
    absolute_low = np.maximum(np.maximum(low, -high), 0.0)
    absolute_high = np.maximum(np.abs(low), np.abs(high))
    lower = np.zeros(low.shape[:-1], dtype=np.float64)
    upper = lower.copy()
    for axis in range(3):
        lower = np.maximum(
            0.0,
            np.nextafter(
                lower
                + np.maximum(0.0, np.nextafter(absolute_low[..., axis] ** 2, -np.inf)),
                -np.inf,
            ),
        )
        upper = np.nextafter(
            upper + np.nextafter(absolute_high[..., axis] ** 2, np.inf), np.inf
        )
    return np.maximum(0.0, np.nextafter(np.sqrt(lower), -np.inf)), np.nextafter(
        np.sqrt(upper), np.inf
    )


def interval_vector_cross(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    j, k = [1, 2, 0], [2, 0, 1]
    lower, upper = interval_subtract(
        interval_multiply(
            (first[0][..., j], first[1][..., j]), (second[0][..., k], second[1][..., k])
        ),
        interval_multiply(
            (first[0][..., k], first[1][..., k]), (second[0][..., j], second[1][..., j])
        ),
    )
    return np.asarray(lower), np.asarray(upper)


def sphere_triangle_distance_bounds(
    frame: tuple[np.ndarray, float, float], corners: np.ndarray, /
) -> np.ndarray:
    """True Euclidean source distance bounds, not a same-UV interpolant norm.

    The full convex source is contained in the radial shell. Triangle interiors
    are contained in the source plus their vertex-distance enclosure by
    convexity. Source-to-mesh coverage additionally needs the independently
    verified closed radial degree-one chain.
    """
    center, radius_low, radius_high = frame
    relative = interval_subtract((corners, corners), (center, center))
    relative = (np.asarray(relative[0]), np.asarray(relative[1]))
    first = interval_subtract(
        (corners[:, 1], corners[:, 1]), (corners[:, 0], corners[:, 0])
    )
    first = (np.asarray(first[0]), np.asarray(first[1]))
    second = interval_subtract(
        (corners[:, 2], corners[:, 2]), (corners[:, 0], corners[:, 0])
    )
    second = (np.asarray(second[0]), np.asarray(second[1]))
    normal = interval_vector_cross(first, second)
    product = interval_multiply(normal, (relative[0][:, 0], relative[1][:, 0]))
    product = (np.asarray(product[0]), np.asarray(product[1]))
    dot_low, dot_high = np.zeros(corners.shape[0]), np.zeros(corners.shape[0])
    for axis in range(3):
        dot_low, dot_high = interval_add(
            (dot_low, dot_high), (product[0][:, axis], product[1][:, axis])
        )
        dot_low, dot_high = np.asarray(dot_low), np.asarray(dot_high)
    normal_high = interval_vector_norm(normal)[1]
    numerator = np.maximum(np.maximum(dot_low, -dot_high), 0.0)
    distance_low = np.nextafter(
        numerator / np.where(normal_high > 0.0, normal_high, 1.0), -np.inf
    )
    repeated01 = np.all(corners[:, 0] == corners[:, 1], axis=1)
    repeated02 = np.all(corners[:, 0] == corners[:, 2], axis=1)
    repeated12 = np.all(corners[:, 1] == corners[:, 2], axis=1)
    collapsed = repeated01 | repeated02 | repeated12
    if np.any(collapsed):
        rows = np.flatnonzero(collapsed)
        endpoint = np.where(repeated01[rows], 2, 1)
        a = (relative[0][rows, 0], relative[1][rows, 0])
        b = (relative[0][rows, endpoint], relative[1][rows, endpoint])
        cross_low = interval_vector_norm(interval_vector_cross(a, b))[0]
        edge = interval_subtract(a, b)
        edge_high = interval_vector_norm((np.asarray(edge[0]), np.asarray(edge[1])))[1]
        segment_low = np.nextafter(
            cross_low / np.where(edge_high > 0.0, edge_high, 1.0), -np.inf
        )
        same = np.all(corners[rows, 0] == corners[rows, endpoint], axis=1)
        segment_low[same] = interval_vector_norm(a)[0][same]
        distance_low[rows] = segment_low
    # The closest point of an obtuse or thin triangle can lie on an edge,
    # outside its supporting plane's orthogonal foot. Additional separating
    # planes give rigorous triangle-specific lower radii without a solve.
    for first, second in ((0, 1), (1, 2), (2, 0)):
        direction = 0.5 * corners[:, first] + 0.5 * corners[:, second] - center
        norm_high = interval_vector_norm((direction, direction))[1]
        products = interval_multiply(
            relative,
            (direction[:, None], direction[:, None]),
        )
        products = (np.asarray(products[0]), np.asarray(products[1]))
        dot_low = np.zeros(corners.shape[:2], dtype=np.float64)
        dot_high = dot_low.copy()
        for axis in range(3):
            dot_low, dot_high = interval_add(
                (dot_low, dot_high),
                (products[0][..., axis], products[1][..., axis]),
            )
            dot_low, dot_high = np.asarray(dot_low), np.asarray(dot_high)
        support = np.nextafter(
            np.min(dot_low, axis=1) / np.where(norm_high > 0.0, norm_high, 1.0),
            -np.inf,
        )
        distance_low = np.maximum(distance_low, np.maximum(support, 0.0))
    node_low, node_high = interval_vector_norm(relative)
    node_error = np.max(
        np.maximum(node_high - radius_low, radius_high - node_low), axis=1
    )
    return np.nextafter(
        np.maximum(np.maximum(radius_high - distance_low, node_error), 0.0), np.inf
    )


def sphere_triangle_normal_bounds(
    frame: tuple[np.ndarray, float, float], corners: np.ndarray, /
) -> np.ndarray:
    """Normal diameter over the radial source pieces of a whole sphere.

    A normalized convex combination of acute unit directions has diameter at
    most their largest pairwise angle. The ellipsoid Gauss map contributes its
    squared axis-condition bound; the shell width encloses center displacement.
    """
    center, radius_low, radius_high = frame
    relative = interval_subtract((corners, corners), (center, center))
    relative = (np.asarray(relative[0]), np.asarray(relative[1]))
    norms = interval_vector_norm(relative)[1]
    cosine = np.ones((corners.shape[0],), dtype=np.float64)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        products = interval_multiply(
            (relative[0][:, first], relative[1][:, first]),
            (relative[0][:, second], relative[1][:, second]),
        )
        products = (np.asarray(products[0]), np.asarray(products[1]))
        low, high = np.zeros(corners.shape[0]), np.zeros(corners.shape[0])
        for axis in range(3):
            low, high = interval_add(
                (low, high), (products[0][:, axis], products[1][:, axis])
            )
            low, high = np.asarray(low), np.asarray(high)
        denominator = np.nextafter(norms[:, first] * norms[:, second], np.inf)
        candidate = np.nextafter(
            low / np.where(denominator > 0, denominator, 1.0), -np.inf
        )
        cosine = np.minimum(cosine, candidate)
    diameter = np.nextafter(np.arccos(np.clip(cosine, 0.0, 1.0)), np.inf)
    diameter = np.where(cosine > 0.0, diameter, np.pi)
    displacement = 2 * np.nextafter(
        np.arcsin(min(1.0, (radius_high - radius_low) / radius_low)), np.inf
    )
    condition = np.nextafter((radius_high / radius_low) ** 2, np.inf)
    return np.minimum(np.pi, np.nextafter(condition * (diameter + displacement), np.inf))


def sphere_radial_degree_one(
    frame: tuple[np.ndarray, float, float],
    points: np.ndarray,
    cells: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> bool:
    """signed ray degree plus closed, consistently radial triangle chain."""

    active = cells[
        ~global_embedding_owner.degenerate_simplices(points, cells, budget=budget)
    ]
    if not active.size:
        return False
    integers, _ = rational_owner.dyadic_integers(np.concatenate((frame[0][None], points)))
    corners = integers[1:][active]
    x = corners - integers[0]
    cross = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    determinant = np.sum(cross * x[:, 0], axis=1)
    positive, negative = determinant > 0, determinant < 0
    if not (np.all(positive) or np.all(negative)):
        return False
    # Geometry equality is checked here, not reconstructed as scientific IDs.
    # Seam/pole aliases already follow the verified source-use provenance.
    directed = {}
    for triangle in points[active]:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = tuple(triangle[first]), tuple(triangle[second])
            if directed.get((b, a), 0):
                directed[(b, a)] -= 1
            else:
                directed[(a, b)] = directed.get((a, b), 0) + 1
    if any(directed.values()):
        return False
    degree, certain = global_embedding_owner.ray_crossings_3d(
        frame[0], points[active], budget=budget
    )
    return certain and abs(degree) == 1


def carrier_breakpoints(
    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
) -> np.ndarray:
    """Preserve source span/chart boundaries under authored affine maps."""
    match curve:
        case PlacedCurve():
            return carrier_breakpoints(curve.definition, first, last)
        case constructors_owner.NormalizedTrimCurve():
            return carrier_breakpoints(curve.curve, first, last)
        case CurveTrimSegment():
            ends = np.asarray(
                (curve.source_parameters(first), curve.source_parameters(last)),
                dtype=np.float64,
            )
            points = carrier_breakpoints(
                curve.curve, float(np.min(ends)), float(np.max(ends))
            )
            local = (points - curve.first) / (curve.last - curve.first)
            return np.sort(1.0 - local if curve.reversed else local)
        case IntersectionPCurve():
            ends = np.asarray(
                (curve._carrier(first), curve._carrier(last)), dtype=np.float64
            )
            points = carrier_breakpoints(
                curve.curve, float(np.min(ends)), float(np.max(ends))
            )
            return np.sort(
                curve.first + curve.last - points if curve.reversed else points
            )
        case IntersectionCurve():
            points = np.arange(np.ceil(first), np.floor(last) + 1, dtype=np.float64)
            return points[(points > first) & (points < last)]
        case BSplineCurve():
            points = np.unique(np.asarray(curve.knots))
            return points[(points > first) & (points < last)]
        case _:
            return np.empty((0,), dtype=np.float64)


def curve_c1_interval(
    curve: AbstractCurve | AbstractTrimCurve | IntersectionCurve,
    first: float,
    last: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
    restriction_bank: BernsteinRestrictionBank | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> bool:
    """Whether a second-jet Taylor bound applies over this entire parameter arc."""

    match curve:
        case PlacedCurve():
            return curve_c1_interval(
                curve.definition,
                first,
                last,
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
        case constructors_owner.NormalizedTrimCurve():
            return curve_c1_interval(
                curve.curve,
                first,
                last,
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
        case CurveTrimSegment():
            # The segment owns endpoint roots and its source chart range. Calling
            # the owner preserves certified source extensions instead of reducing
            # them to rounded carrier endpoints.
            return curve.is_c1_on(first, last, budget=budget, preparations=preparations)
        case IntersectionPCurve():
            ends = sorted((float(curve._carrier(first)), float(curve._carrier(last))))
            return curve_c1_interval(
                curve.curve,
                ends[0],
                ends[1],
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
        case IntersectionCurve():
            # Closed source intervals touching an atlas node include both
            # one-sided graph velocities. Shared nodes prove C0, not C1; use the
            # jet admission rather than a half-open chart-index guess.
            return curve.fully_certified and curve.is_c1_on(
                first, last, budget=budget, preparations=preparations
            )
        case SurfaceIsoparametricCurve():
            return curve.is_c1_on(first, last, budget=budget, preparations=preparations)
        case BSplineCurve():
            # Closed intervals touching a C0 knot (multiplicity >= degree) are
            # refused by the order-two jet; admit exactly as it does so

            # the first-jet remainder, not an infinite second jet, bounds the arc.
            return curve.is_c1_on(first, last, budget=budget, preparations=preparations)
        case LineCurve() | CircleCurve() | EllipseCurve():
            return True
        case _:
            return False


def first_jet_chord_remainder(
    lower: np.ndarray,
    upper: np.ndarray,
    width: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Continuous piecewise-source interpolation from its whole first jet.

    Subtract the midpoint Jacobian, whose affine contribution cancels. Each
    derivative column then has Lipschitz constant half its interval diameter;
    barycentric mean absolute deviation is at most half the chart width.
    This is the same integral remainder used by the source extrusion owner,
    and does not assert a Hessian or C1 continuity across a source knot.
    """

    budget.reserve(4 * lower.size, 256 + 32 * lower.size + 64 * width.size)
    coordinate_enclosure_owner.reserve_polynomial(
        0, 2 * width.size, 0, 2200, budget=budget
    )
    diameter = np.asarray(interval_subtract((upper, upper), (lower, lower))[1])
    terms = []
    for axis, extent in enumerate(width):
        if extent == 0:
            continue
        norm = source_interpolation_owner.norm_upper(diameter[:, axis], budget=budget)
        if not np.isfinite(norm) or not np.isfinite(extent) or extent < 0:
            return np.inf
        terms.append((Fraction(norm), Fraction(float(np.nextafter(extent, np.inf)))))
    planar_primitives_owner.reserve_fraction_work(
        tuple(terms), 3 * len(terms) + 1, 1, 2 * len(terms) + 1, budget=budget
    )
    bound = sum((norm * extent / 4 for norm, extent in terms), Fraction())
    if bound > Fraction(float(np.finfo(np.float64).max)):
        return np.inf
    value = float(bound)
    return value if Fraction(value) >= bound else float(np.nextafter(value, np.inf))


def continuous_spline_carrier(
    surface: AbstractSurfacePatch, /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Whether a validated C0 spline carrier can require first-jet bounds.

    An offset is continuous across a base knot only where the base unit normal
    is; it is admitted only over a planar-meridian revolution, whose offset
    jets prove G1 at every straddled junction and stay unbounded otherwise.
    """
    while isinstance(surface, PlacedSurface):
        surface = surface.definition
    if isinstance(surface, OffsetSurface):
        return (
            isinstance(surface.base, RevolutionSurface)
            and meridian_frame(surface.base, budget=budget) is not None
        )
    return isinstance(
        surface,
        (BSplineSurfacePatch, ExtrusionSurface, RevolutionSurface, RuledSurface),
    )


def curve_chord_error(
    curve: AbstractTrimCurve,
    first: float,
    last: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
    restriction_bank: BernsteinRestrictionBank | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> np.ndarray:
    """Continuous chord error, including legal first-jet jumps at source spans."""
    if curve_c1_interval(
        curve,
        first,
        last,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    ):
        lower, upper = curve.derivative_bounds(
            first,
            last,
            order=2,
            budget=budget,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        return np.maximum(np.abs(lower), np.abs(upper)) * (last - first) ** 2 / 8
    lower, upper = curve.derivative_bounds(
        first,
        last,
        order=1,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )
    # Integral remainders with f' in [lower, upper] give t(1-t) Δf' Δs.
    # Unlike an order-two union, this remains valid across proved C0 joins.
    return np.nextafter(upper - lower, np.inf) * (last - first) / 4
