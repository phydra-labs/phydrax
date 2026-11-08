#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Whole-edge interval evidence for exact source edge classification."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
)
from ...typing import ConvertibleToArray, parse
from .._interval_enclosure import (
    Interval,
    interval_add,
    interval_divide,
    interval_multiply,
    interval_sqrt,
    interval_square,
    interval_subtract,
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import fraction_down, fraction_up
from ._model import BRepEntityId, BRepGeometry, BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    SurfaceIsoparametricCurve,
)


type BRepEdgeStatus = Literal["convex", "concave", "sign_change", "unresolved"]


def _cross(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    components = []
    for left, right in ((1, 2), (2, 0), (0, 1)):
        components.append(
            interval_subtract(
                interval_multiply(
                    (first[0][left], first[1][left]), (second[0][right], second[1][right])
                ),
                interval_multiply(
                    (first[0][right], first[1][right]), (second[0][left], second[1][left])
                ),
            )
        )
    return np.asarray([item[0] for item in components]), np.asarray(
        [item[1] for item in components]
    )


def _dot(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], /
) -> tuple[float, float]:
    result = (np.asarray(0.0), np.asarray(0.0))
    for axis in range(first[0].shape[0]):
        result = interval_add(
            result,
            interval_multiply(
                (first[0][axis], first[1][axis]),
                (second[0][axis], second[1][axis]),
            ),
        )
    return float(result[0]), float(result[1])


def _norm(value: tuple[np.ndarray, np.ndarray], /) -> tuple[float, float]:
    result = (np.asarray(0.0), np.asarray(0.0))
    for axis in range(value[0].shape[0]):
        result = interval_add(
            result,
            interval_square((value[0][axis], value[1][axis])),
        )
    length = interval_sqrt(result)
    return float(length[0]), float(length[1])


def _material_orientations(model: BRepModel, /) -> np.ndarray:
    orientations = np.asarray(model.orientation, dtype=np.float64).copy()
    memberships: list[set[int]] = [set() for _ in model.patches]
    for faces, signs in zip(
        model.topology.solid_faces,
        model.topology.solid_face_orientations,
        strict=True,
    ):
        for face, sign in zip(faces, signs, strict=True):
            memberships[face].add(sign)
    for face, signs in enumerate(memberships):
        if len(signs) > 1:
            orientations[face] = np.nan
        elif signs:
            orientations[face] *= next(iter(signs))
    return orientations


def _coedge_face_owners(geometry: BRepGeometry, /) -> tuple[int, ...]:
    owners = [-1] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                if owners[coedge] not in (-1, face):
                    raise ValueError("A source coedge belongs to more than one face.")
                owners[coedge] = face
    if any(face < 0 for face in owners):
        raise ValueError("Every source coedge must belong to exactly one face.")
    return tuple(owners)


def _prepare(
    function: Callable[[Array], Array], dimension: int, /
) -> PreparedIntervalFunction | None:
    try:
        return prepare_interval_function(function, dimension)
    except (IntervalExtensionError, NotImplementedError, TypeError, ValueError):
        return None


def _curve_point_function(curve: AbstractCurve, /) -> Callable[[Array], Array]:
    def evaluate(parameter: Array, /) -> Array:
        return curve.evaluate(parameter[0])

    return evaluate


def _curve_tangent_function(curve: AbstractCurve, /) -> Callable[[Array], Array]:
    def evaluate(parameter: Array, /) -> Array:
        return jax.jacfwd(curve.evaluate)(parameter[0])

    return evaluate


def _evaluate(
    prepared: PreparedIntervalFunction | None,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    if prepared is None:
        return None
    try:
        low, high = prepared.evaluate(lower.reshape((1, -1)), upper.reshape((1, -1)))
    except (
        IntervalExtensionError,
        NotImplementedError,
        TypeError,
        ValueError,
        FloatingPointError,
    ):
        return None
    low, high = (
        np.asarray(low[0], dtype=np.float64),
        np.asarray(high[0], dtype=np.float64),
    )
    if np.any(np.isnan(low)) or np.any(np.isnan(high)) or np.any(low > high):
        return None
    return low, high


def _curve_derivative_bounds(
    curve: AbstractCurve,
    prepared: PreparedIntervalFunction | None,
    first: float,
    last: float,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    if type(curve).derivative_bounds is not AbstractCurve.derivative_bounds:
        try:
            lower, upper = curve.derivative_bounds(first, last)
        except (
            IntervalExtensionError,
            NotImplementedError,
            TypeError,
            ValueError,
            FloatingPointError,
        ):
            pass
        else:
            lower_, upper_ = (
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
            )
            if (
                lower_.shape == (curve.ambient_dimension,)
                and upper_.shape == lower_.shape
                and not np.any(np.isnan(lower_))
                and not np.any(np.isnan(upper_))
                and not np.any(lower_ > upper_)
            ):
                return lower_, upper_
    if isinstance(curve, SurfaceIsoparametricCurve):
        lower = np.empty((2,), dtype=np.float64)
        upper = np.empty((2,), dtype=np.float64)
        free_axis = 1 - curve.fixed_axis
        lower[curve.fixed_axis] = upper[curve.fixed_axis] = curve.fixed_value
        lower[free_axis], upper[free_axis] = first, last
        derivatives = _surface_jacobian_bounds(curve.surface, None, lower, upper)
        if derivatives is not None and derivatives[0].shape == (3, 2):
            return derivatives[0][:, free_axis], derivatives[1][:, free_axis]
    return _evaluate(prepared, np.asarray((first,)), np.asarray((last,)))


def _surface_jacobian_bounds(
    surface: AbstractSurfacePatch,
    prepared: PreparedIntervalFunction | None,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    if type(surface).derivative_bounds is not AbstractSurfacePatch.derivative_bounds:
        lower_box = np.array(lower, dtype=np.float64, copy=True)
        upper_box = np.array(upper, dtype=np.float64, copy=True)
        domain = surface.parameter_domain
        for axis in range(2):
            if lower_box[axis] != upper_box[axis]:
                continue
            value = lower_box[axis]
            if domain is not None and value <= domain[0][axis]:
                upper_box[axis] = np.nextafter(value, np.inf)
            elif domain is not None and value >= domain[1][axis]:
                lower_box[axis] = np.nextafter(value, -np.inf)
            else:
                lower_box[axis] = np.nextafter(value, -np.inf)
                upper_box[axis] = np.nextafter(value, np.inf)
        try:
            low, high = surface.derivative_bounds(np.stack((lower_box, upper_box)))
        except (
            IntervalExtensionError,
            NotImplementedError,
            TypeError,
            ValueError,
            FloatingPointError,
        ):
            pass
        else:
            low_, high_ = (
                np.asarray(low, dtype=np.float64),
                np.asarray(high, dtype=np.float64),
            )
            if (
                low_.shape == (3, 2)
                and high_.shape == low_.shape
                and not np.any(np.isnan(low_))
                and not np.any(np.isnan(high_))
                and not np.any(low_ > high_)
            ):
                return low_, high_
    return _evaluate(prepared, lower, upper)


def _source_curve_box(
    curve: AbstractCurve,
    prepared: PreparedIntervalFunction | None,
    first: float,
    last: float,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    if type(curve).bounding_box is not AbstractCurve.bounding_box:
        try:
            box = np.asarray(curve.bounding_box(first, last), dtype=np.float64)
        except (
            IntervalExtensionError,
            NotImplementedError,
            TypeError,
            ValueError,
            FloatingPointError,
        ):
            return None
        if (
            box.shape != (2, curve.ambient_dimension)
            or not np.all(np.isfinite(box))
            or np.any(box[1] < box[0])
        ):
            return None
        return box[0], box[1]
    return _evaluate(prepared, np.asarray((first,)), np.asarray((last,)))


def _normal(
    pcurve_source: AbstractCurve,
    pcurve: PreparedIntervalFunction | None,
    surface: AbstractSurfacePatch,
    surface_jacobian: PreparedIntervalFunction | None,
    first: float,
    last: float,
    orientation: float,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    uv = _source_curve_box(pcurve_source, pcurve, first, last)
    if uv is None:
        return None
    derivatives = _surface_jacobian_bounds(surface, surface_jacobian, uv[0], uv[1])
    if derivatives is None or derivatives[0].shape != (3, 2):
        return None
    normal = _cross(
        (derivatives[0][:, 0], derivatives[1][:, 0]),
        (derivatives[0][:, 1], derivatives[1][:, 1]),
    )
    if orientation < 0:
        return -normal[1], -normal[0]
    return normal


def _float_interval(bounds: tuple[float, float], /) -> Interval:
    return np.asarray(bounds[0]), np.asarray(bounds[1])


def _cell_sign_angle(
    edge_curve: AbstractCurve,
    edge_curve_plan: PreparedIntervalFunction | None,
    pcurve_sources: tuple[AbstractCurve, AbstractCurve],
    surfaces: tuple[AbstractSurfacePatch, AbstractSurfacePatch],
    patch_pcurves: tuple[
        PreparedIntervalFunction | None, PreparedIntervalFunction | None
    ],
    surface_jacobians: tuple[
        PreparedIntervalFunction | None, PreparedIntervalFunction | None
    ],
    orientations: tuple[float, float],
    tangent_sign: float,
    first: float,
    last: float,
    /,
) -> tuple[float, float, float, float, bool]:
    normal_a = _normal(
        pcurve_sources[0],
        patch_pcurves[0],
        surfaces[0],
        surface_jacobians[0],
        first,
        last,
        orientations[0],
    )
    normal_b = _normal(
        pcurve_sources[1],
        patch_pcurves[1],
        surfaces[1],
        surface_jacobians[1],
        first,
        last,
        orientations[1],
    )
    tangent_value = _curve_derivative_bounds(edge_curve, edge_curve_plan, first, last)
    if (
        normal_a is None
        or normal_b is None
        or tangent_value is None
        or tangent_value[0].shape != (3,)
    ):
        return -np.inf, np.inf, 0.0, float(np.nextafter(np.pi, np.inf)), False
    tangent = (tangent_value[0], tangent_value[1])
    if tangent_sign < 0:
        tangent = (-tangent[1], -tangent[0])
    sign = _dot(normal_a, _cross(normal_b, tangent))
    norm_a, norm_b, norm_tangent = _norm(normal_a), _norm(normal_b), _norm(tangent)
    if not (norm_a[0] > 0.0 and norm_b[0] > 0.0 and norm_tangent[0] > 0.0):
        return sign[0], sign[1], 0.0, float(np.nextafter(np.pi, np.inf)), False
    dot = _dot(normal_a, normal_b)
    if dot[0] > 0.0:
        ratio = interval_divide(
            _float_interval(_norm(_cross(normal_a, normal_b))),
            _float_interval(dot),
        )
        lower = float(np.nextafter(np.arctan(max(0.0, float(ratio[0]))), -np.inf))
        upper = float(np.nextafter(np.arctan(max(0.0, float(ratio[1]))), np.inf))
    else:
        cosine = interval_divide(
            _float_interval(dot),
            interval_multiply(_float_interval(norm_a), _float_interval(norm_b)),
        )
        lower = float(
            np.nextafter(np.arccos(np.clip(float(cosine[1]), -1.0, 1.0)), -np.inf)
        )
        upper = float(
            np.nextafter(np.arccos(np.clip(float(cosine[0]), -1.0, 1.0)), np.inf)
        )
    return (
        sign[0],
        sign[1],
        max(0.0, lower),
        min(float(np.nextafter(np.pi, np.inf)), upper),
        True,
    )


def _coedge_source_lift(
    model: BRepModel,
    geometry: BRepGeometry,
    coedge: int,
    face: int,
    /,
) -> bool:
    edge = geometry.coedge_edges[coedge]
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        return False
    curve = geometry.curves[curve_index]
    pcurve = geometry.pcurves[coedge]
    patch = model.patches[face]
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges[edge]))
    if _isoparametric_source_lift(curve, pcurve, patch):
        return True
    from ._correspondence import exact_source_lift
    from ._intersection_curve import SurfaceRegion

    region_box = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    try:
        region = SurfaceRegion(patch, region_box)
        if exact_source_lift(curve, pcurve, region, first, last):
            return True
        from ._conic_gaps import native_face_gap_squared

        return native_face_gap_squared(curve, pcurve, region, first, last) == 0
    except (NotImplementedError, TypeError, ValueError, FloatingPointError):
        return False


def _isoparametric_source_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: AbstractSurfacePatch,
    /,
) -> bool:
    if not isinstance(curve, SurfaceIsoparametricCurve) or curve.surface is not patch:
        return False
    from ._source_correspondence import _affine_curve_coefficients

    coefficients = _affine_curve_coefficients(pcurve, (Fraction(0), Fraction(0)))
    if coefficients is None:
        return False
    origin, direction = coefficients
    fixed_axis = curve.fixed_axis
    free_axis = 1 - fixed_axis
    return (
        origin[fixed_axis] == Fraction.from_float(curve.fixed_value)
        and origin[free_axis] == 0
        and direction[fixed_axis] == 0
        and direction[free_axis] == 1
    )


def _coedge_path_c1(
    pcurve: AbstractCurve,
    patch: AbstractSurfacePatch,
    prepared: PreparedIntervalFunction | None,
    first: float,
    last: float,
    /,
) -> bool:
    try:
        if not pcurve.is_c1_on(first, last):
            return False
        uv = _source_curve_box(pcurve, prepared, first, last)
        if uv is None or uv[0].shape != (2,):
            return False
        fixed_axes = np.flatnonzero(uv[0] == uv[1])
        if fixed_axes.size:
            fixed_axis = int(fixed_axes[0])
            free_axis = 1 - fixed_axis
            return bool(
                patch.is_c1_along(
                    fixed_axis,
                    float(uv[0][fixed_axis]),
                    float(uv[0][free_axis]),
                    float(uv[1][free_axis]),
                )
            )
        return bool(patch.is_c1_on(np.stack((uv[0], uv[1]))))
    except (IntervalExtensionError, NotImplementedError, TypeError, ValueError):
        return False


def _edge_status(signs: list[tuple[float, float]], regular: bool, /) -> BRepEdgeStatus:
    if not regular or not signs:
        return "unresolved"
    if all(upper < 0.0 for _, upper in signs):
        return "convex"
    if all(lower > 0.0 for lower, _ in signs):
        return "concave"
    if any(upper < 0.0 for _, upper in signs) and any(lower > 0.0 for lower, _ in signs):
        return "sign_change"
    return "unresolved"


@final
class BRepEdgeClassification(StrictModule):
    """Whole-edge source interval classification and dihedral enclosures."""

    edge_id: BRepEntityId = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    status: BRepEdgeStatus = eqx.field(static=True)
    change_parameters: Array
    dihedral_range: Array
    length_enclosure: Array
    sign_enclosure: Array

    def __init__(
        self,
        edge_id: BRepEntityId,
        model_id: str,
        status: BRepEdgeStatus,
        change_parameters: ConvertibleToArray,
        dihedral_range: ConvertibleToArray,
        length_enclosure: ConvertibleToArray,
        sign_enclosure: ConvertibleToArray,
        /,
    ) -> None:
        if not isinstance(edge_id, BRepEntityId) or edge_id.kind != "edge":
            raise TypeError("BRepEdgeClassification.edge_id must identify a source edge.")
        status_ = parse(status, BRepEdgeStatus, "status")
        self.edge_id = edge_id
        self.model_id = model_id
        self.status = status_
        self.change_parameters = jnp.asarray(
            change_parameters, dtype=jnp.float64
        ).reshape((-1, 2))
        self.dihedral_range = jnp.asarray(dihedral_range, dtype=jnp.float64).reshape((2,))
        self.length_enclosure = jnp.asarray(length_enclosure, dtype=jnp.float64).reshape(
            (2,)
        )
        self.sign_enclosure = jnp.asarray(sign_enclosure, dtype=jnp.float64).reshape((2,))


def _exact_revolution_circle_classification(
    model: BRepModel,
    geometry: BRepGeometry,
    edge: int,
    coedges: tuple[int, ...],
    faces: tuple[int, int],
    orientations: np.ndarray,
    /,
) -> BRepEdgeClassification | None:
    from ._correspondence import exact_source_lift
    from ._intersection import NativePeriodEndpoint
    from ._intersection_curve import SurfaceRegion
    from ._patches import (
        CircleCurve,
        RevolutionQuadricWitness,
        RevolutionSurface,
    )
    from ._trim_helpers import period_scalar_bounds, PeriodOffset

    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        return None
    curve = geometry.curves[curve_index]
    if (
        not isinstance(curve, CircleCurve)
        or geometry.edge_vertices[edge][0] != geometry.edge_vertices[edge][1]
    ):
        return None
    first_root, last_root = geometry.edge_endpoint_roots[edge]
    if not (
        isinstance(first_root, NativePeriodEndpoint)
        and isinstance(last_root, NativePeriodEndpoint)
        and first_root.curve is curve
        and last_root.curve is curve
        and first_root.rational == last_root.rational == 0
        and first_root.turns == 0
        and last_root.turns == 1
    ):
        return None

    witnesses: list[RevolutionQuadricWitness] = []
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    for coedge, face in zip(coedges, faces, strict=True):
        patch = model.patches[face]
        if not isinstance(patch, RevolutionSurface):
            return None
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        witness = patch.exact_quadric_witness((float(bounds[0, 1]), float(bounds[1, 1])))
        if witness is None:
            return None
        if (
            not isinstance(witness.profile_parameter_interval, tuple)
            or len(witness.profile_parameter_interval) != 2
        ):
            return None
        pcurve = geometry.pcurves[coedge]
        if not exact_source_lift(
            curve, pcurve, SurfaceRegion(patch, bounds), first, last
        ):
            return None
        witnesses.append(witness)

    supported_families = ("plane", "cylinder", "cone", "sphere")
    if any(witness.family not in supported_families for witness in witnesses):
        return None
    if witnesses[0].family == witnesses[1].family:
        return None
    axis = witnesses[0].axis_exact
    radial_axis = witnesses[0].radial_axis_exact
    second_radial_axis = witnesses[0].second_radial_axis_exact
    if not (
        witnesses[1].axis_exact == axis
        and witnesses[1].radial_axis_exact == radial_axis
        and witnesses[1].second_radial_axis_exact == second_radial_axis
        and curve.first_axis_exact == radial_axis
        and curve.second_axis_exact == second_radial_axis
    ):
        return None

    def dot(
        first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum(
            (left * right for left, right in zip(first, second, strict=True)),
            Fraction(0),
        )

    def cross(
        first: tuple[ExactScalar, ExactScalar, ExactScalar],
        second: tuple[ExactScalar, ExactScalar, ExactScalar],
        /,
    ) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    point: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        curve.center_exact[0] + curve.radius_exact * curve.first_axis_exact[0],
        curve.center_exact[1] + curve.radius_exact * curve.first_axis_exact[1],
        curve.center_exact[2] + curve.radius_exact * curve.first_axis_exact[2],
    )
    if dot(point, point) == 0:
        return None
    center = curve.center_exact
    for witness in witnesses:
        axis_origin = (
            witness.center_exact if witness.family == "sphere" else witness.origin_exact
        )
        if axis_origin is None:
            return None
        offset = tuple(
            value - base for value, base in zip(center, axis_origin, strict=True)
        )
        axial = dot(offset, axis)
        if offset != tuple(axial * value for value in axis):
            return None

    def surface_normal(
        witness: RevolutionQuadricWitness, /
    ) -> tuple[ExactScalar, ExactScalar, ExactScalar] | None:
        family = witness.family
        if family == "sphere":
            if witness.center_exact is None or witness.radius_exact in (None, 0):
                return None
            return (
                witness.frame_sign * (point[0] - witness.center_exact[0]),
                witness.frame_sign * (point[1] - witness.center_exact[1]),
                witness.frame_sign * (point[2] - witness.center_exact[2]),
            )
        radial_direction = witness.radial_direction_exact
        axial_direction = witness.axial_direction_exact
        if radial_direction is None or axial_direction is None:
            return None
        radial_value = curve.radius_exact
        return (
            witness.frame_sign
            * radial_value
            * (
                axial_direction * witness.radial_axis_exact[0]
                - radial_direction * witness.axis_exact[0]
            ),
            witness.frame_sign
            * radial_value
            * (
                axial_direction * witness.radial_axis_exact[1]
                - radial_direction * witness.axis_exact[1]
            ),
            witness.frame_sign
            * radial_value
            * (
                axial_direction * witness.radial_axis_exact[2]
                - radial_direction * witness.axis_exact[2]
            ),
        )

    raw_normals = tuple(surface_normal(witness) for witness in witnesses)
    normal_a_source, normal_b_source = raw_normals
    if normal_a_source is None or normal_b_source is None:
        return None
    normal_a = (
        int(orientations[faces[0]]) * normal_a_source[0],
        int(orientations[faces[0]]) * normal_a_source[1],
        int(orientations[faces[0]]) * normal_a_source[2],
    )
    normal_b = (
        int(orientations[faces[1]]) * normal_b_source[0],
        int(orientations[faces[1]]) * normal_b_source[1],
        int(orientations[faces[1]]) * normal_b_source[2],
    )
    tangent_sign = int(geometry.coedge_senses[coedges[1]]) * int(orientations[faces[1]])
    tangent: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        tangent_sign * curve.radius_exact * curve.second_axis_exact[0],
        tangent_sign * curve.radius_exact * curve.second_axis_exact[1],
        tangent_sign * curve.radius_exact * curve.second_axis_exact[2],
    )
    sign = dot(normal_a, cross(normal_b, tangent))
    if sign == 0:
        return None

    def scalar_interval(value: ExactScalar, /) -> Interval:
        lower, upper = coefficient_interval(value, bits=128)
        return np.asarray(fraction_down(lower)), np.asarray(fraction_up(upper))

    sign_bounds = (
        float(scalar_interval(sign)[0]),
        float(scalar_interval(sign)[1]),
    )
    if not (sign_bounds[1] < 0.0 or sign_bounds[0] > 0.0):
        return None
    normal_product = interval_multiply(
        scalar_interval(dot(normal_a, normal_a)),
        scalar_interval(dot(normal_b, normal_b)),
    )
    norm_product = interval_sqrt(normal_product)
    cosine = interval_divide(scalar_interval(dot(normal_a, normal_b)), norm_product)
    if not np.all(np.isfinite(cosine)):
        return None
    cosine_lower = float(np.clip(cosine[0], -1.0, 1.0))
    cosine_upper = float(np.clip(cosine[1], -1.0, 1.0))
    dihedral = (
        max(0.0, float(np.nextafter(np.arccos(cosine_upper), -np.inf))),
        min(
            float(np.nextafter(np.pi, np.inf)),
            float(np.nextafter(np.arccos(cosine_lower), np.inf)),
        ),
    )
    radius_bounds = scalar_interval(curve.radius_exact)
    period_bounds = period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(1)))
    length = interval_multiply(radius_bounds, period_bounds)
    length_enclosure = (max(0.0, float(length[0])), float(length[1]))
    if not all(np.isfinite(value) for value in (*dihedral, *length_enclosure)):
        return None
    status = _edge_status([sign_bounds], True)
    return BRepEdgeClassification(
        model.edge_ids[edge],
        model.model_id,
        status,
        np.empty((0, 2), dtype=np.float64),
        np.asarray(dihedral),
        np.asarray(length_enclosure),
        np.asarray(sign_bounds),
    )


def _edge_classification(
    model: BRepModel,
    geometry: BRepGeometry,
    edge: int,
    coedges: tuple[int, ...],
    owners: tuple[int, ...],
    orientations: np.ndarray,
    patch_plans: tuple[PreparedIntervalFunction | None, ...],
    pcurve_plans: tuple[PreparedIntervalFunction | None, ...],
    curve_plans: tuple[PreparedIntervalFunction | None, ...],
    /,
) -> BRepEdgeClassification:
    curve_index = geometry.edge_curves[edge]
    edge_curve = None if curve_index < 0 else geometry.curves[curve_index]
    edge_plan = None if curve_index < 0 else curve_plans[curve_index]
    first, last = (
        float(value) for value in np.asarray(geometry.edge_ranges[edge], dtype=np.float64)
    )
    sign_cells: list[tuple[float, float]] = []
    angle_cells: list[tuple[float, float]] = []
    changes: list[tuple[float, float]] = []
    length_lower = 0.0
    length_upper = 0.0
    regular = len(coedges) == 2 and edge_curve is not None and first < last
    face_pair: tuple[int, int] | None = None
    face_orientations: tuple[float, float] = (np.nan, np.nan)
    tangent_sign = 1.0
    pair_pcurve_sources: tuple[AbstractCurve, AbstractCurve] | None = None
    pair_surface_sources: tuple[AbstractSurfacePatch, AbstractSurfacePatch] | None = None
    pair_pcurves: tuple[
        PreparedIntervalFunction | None, PreparedIntervalFunction | None
    ] = (None, None)
    pair_surfaces: tuple[
        PreparedIntervalFunction | None, PreparedIntervalFunction | None
    ] = (None, None)
    if regular:
        face_pair = (owners[coedges[0]], owners[coedges[1]])
        face_orientations = (
            float(orientations[face_pair[0]]),
            float(orientations[face_pair[1]]),
        )
        memberships = tuple(model.topology.face_solids[face] for face in face_pair)
        regular = (
            face_pair[0] != face_pair[1]
            and len(memberships[0]) == 1
            and memberships[0] == memberships[1]
            and all(np.isfinite(value) for value in face_orientations)
        )
        if regular and edge_curve is not None:
            from ._intersection_curve import IntersectionCurve

            try:
                regular = bool(edge_curve.is_c1_on(first, last))
                if isinstance(edge_curve, IntersectionCurve):
                    regular = regular and edge_curve.fully_certified
            except (NotImplementedError, TypeError, ValueError):
                regular = False
        if regular and face_pair is not None and edge_curve is not None:
            pair_pcurves = (pcurve_plans[coedges[0]], pcurve_plans[coedges[1]])
            pair_pcurve_sources = (
                geometry.pcurves[coedges[0]],
                geometry.pcurves[coedges[1]],
            )
            pair_surfaces = (patch_plans[face_pair[0]], patch_plans[face_pair[1]])
            pair_surface_sources = (
                model.patches[face_pair[0]],
                model.patches[face_pair[1]],
            )
            tangent_sign = (
                float(geometry.coedge_senses[coedges[1]]) * face_orientations[1]
            )
            shell_directions = tuple(
                float(geometry.coedge_senses[coedges[index]]) * face_orientations[index]
                for index in range(2)
            )
            regular = (
                shell_directions[0] == -shell_directions[1]
                and all(plan is not None for plan in (*pair_pcurves, *pair_surfaces))
                and all(
                    _coedge_source_lift(model, geometry, coedges[index], face_pair[index])
                    and _coedge_path_c1(
                        geometry.pcurves[coedges[index]],
                        model.patches[face_pair[index]],
                        pair_pcurves[index],
                        first,
                        last,
                    )
                    for index in range(2)
                )
            )
    if regular and face_pair is not None and edge_curve is not None:
        exact_classification = _exact_revolution_circle_classification(
            model, geometry, edge, coedges, face_pair, orientations
        )
        if exact_classification is not None:
            return exact_classification

    pending: list[tuple[float, float, tuple[float, float] | None]] = (
        [(first, last, None)] if first < last else []
    )
    while pending:
        start, end, parent_sign_bounds = pending.pop()
        if (
            regular
            and face_pair is not None
            and edge_curve is not None
            and pair_pcurve_sources is not None
            and pair_surface_sources is not None
        ):
            low, high, angle_low, angle_high, valid = _cell_sign_angle(
                edge_curve,
                edge_plan,
                pair_pcurve_sources,
                pair_surface_sources,
                pair_pcurves,
                pair_surfaces,
                face_orientations,
                tangent_sign,
                start,
                end,
            )
        else:
            low, high, angle_low, angle_high, valid = (
                -np.inf,
                np.inf,
                0.0,
                float(np.nextafter(np.pi, np.inf)),
                False,
            )
        unresolved = not valid
        ambiguous = low < 0.0 < high or low == 0.0 or high == 0.0
        midpoint = start + (end - start) * 0.5
        min_subnormal = float(np.nextafter(0.0, np.inf))
        # Noncontracting sign intervals remain unresolved rather than splitting
        # into further cells with identical sign evidence.
        if (
            not unresolved
            and ambiguous
            and max(abs(low), abs(high)) > min_subnormal
            and midpoint > start
            and midpoint < end
            and (
                parent_sign_bounds is None
                or low > parent_sign_bounds[0]
                or high < parent_sign_bounds[1]
            )
        ):
            current_sign_bounds = (low, high)
            pending.append((midpoint, end, current_sign_bounds))
            pending.append((start, midpoint, current_sign_bounds))
            continue
        if unresolved:
            regular = False
            changes.append((start, end))
        elif ambiguous:
            changes.append((start, end))
        sign_cells.append((low, high))
        angle_cells.append((angle_low, angle_high))
        if edge_curve is None:
            length_upper = np.inf
            continue
        speed = _curve_derivative_bounds(edge_curve, edge_plan, start, end)
        if speed is None or speed[0].shape != (3,):
            length_upper = np.inf
            continue
        speed_norm = _norm(speed)
        width = end - start
        width_bounds = (
            max(0.0, float(np.nextafter(width, -np.inf))),
            float(np.nextafter(width, np.inf)),
        )
        contribution = interval_multiply(
            _float_interval(speed_norm), _float_interval(width_bounds)
        )
        length_lower = float(
            np.nextafter(length_lower + max(0.0, float(contribution[0])), -np.inf)
        )
        length_upper = float(np.nextafter(length_upper + float(contribution[1]), np.inf))
    status = _edge_status(sign_cells, regular)
    if sign_cells:
        sign_enclosure = (
            min(item[0] for item in sign_cells),
            max(item[1] for item in sign_cells),
        )
        dihedral = (
            min(item[0] for item in angle_cells),
            max(item[1] for item in angle_cells),
        )
    else:
        sign_enclosure, dihedral = (
            (-np.inf, np.inf),
            (0.0, float(np.nextafter(np.pi, np.inf))),
        )
    return BRepEdgeClassification(
        model.edge_ids[edge],
        model.model_id,
        status,
        np.asarray(changes, dtype=np.float64).reshape((-1, 2)),
        np.asarray(dihedral),
        np.asarray((max(0.0, length_lower), length_upper)),
        np.asarray(sign_enclosure),
    )


def brep_edge_classification(
    model: BRepModel,
    geometry: BRepGeometry,
    /,
) -> tuple[BRepEdgeClassification, ...]:
    """Classify every source edge using whole-range outward interval evidence.

    Ambiguous cells are bisected while their sign enclosures contract until sign
    is proven or binary64 parameter resolution is exhausted. Noncontracting cells
    remain explicitly unresolved.
    """
    if model.geometry is not geometry:
        raise ValueError("BRepEdgeClassification.geometry must belong to the model.")
    orientations = _material_orientations(model)
    owners = _coedge_face_owners(geometry)
    patch_plans = tuple(
        _prepare(jax.jacfwd(patch.evaluate), 2) for patch in model.patches
    )
    pcurve_plans = tuple(
        _prepare(_curve_point_function(pcurve), 1) for pcurve in geometry.pcurves
    )
    curve_plans = tuple(
        _prepare(_curve_tangent_function(curve), 1) for curve in geometry.curves
    )
    uses: list[list[int]] = [[] for _ in geometry.edge_curves]
    for coedge, edge in enumerate(geometry.coedge_edges):
        uses[edge].append(coedge)
    return tuple(
        _edge_classification(
            model,
            geometry,
            edge,
            tuple(uses[edge]),
            owners,
            orientations,
            patch_plans,
            pcurve_plans,
            curve_plans,
        )
        for edge in range(len(geometry.edge_curves))
    )


__all__ = ["BRepEdgeClassification", "BRepEdgeStatus", "brep_edge_classification"]
