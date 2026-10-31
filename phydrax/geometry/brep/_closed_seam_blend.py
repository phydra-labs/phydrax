#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native circular plane/quadric blends, with analytic meridian certificates.

Setbacks are ordered (plane distance, curved-face meridian arclength). Admission
requires a full native turn, a single rectangular quadric face and circular
plane loops. No sampled geometry or closeness threshold determines admission.
The retained plane and quadric carriers survive unchanged; only their trims
and the meridian seam are rebuilt. Binary execution coefficients are accompanied
by outward tangency bounds, rather than being called unrounded mathematical roots.
"""

from __future__ import annotations

from fractions import Fraction
from math import atan2, pi
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from .._interval_enclosure import prepare_interval_function
from .._rational import fraction_down, fraction_up, rational_vector
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ._assembly import BRepAffinePlacement
from ._boolean_core import AffinePCurve, PeriodicPCurve
from ._constructors import _Builder, assemble_brep_model, BRepTessellationPolicy
from ._differentiable import (
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
)
from ._exact import exact_frame_orientation
from ._fillet import BRepBlendUnsupported
from ._intersection import (
    ConicParameterEndpoint,
    NativePeriodEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ._intersection_curve import SurfaceRegion
from ._model import BRepGeometry, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import (
    AbstractCurve,
    BRepCurve,
    CircleCurve,
    ConePatch,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
    RevolutionQuadricWitness,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
)
from ._root_bindings import (
    BRepConicVertex,
    BRepCurveSurfaceLift,
    BRepRootSupport,
    BRepVertexRoot,
)
from ._source_correspondence import _affine_curve_coefficients


@final
class ClosedSeamTangencyCertificate(StrictModule):
    """Whole-circle normal mismatch bounds for the represented torus coefficients.

    The meridian construction solves signed-distance equations analytically.
    ``normal_difference_bounds`` encloses each component of the represented
    boundary normal difference in the orthonormal (radial, axial) frame. It is
    not a tolerance, acceptance threshold, or sampled tangency observation.
    """

    normal_difference_bounds: Array
    radius: float = eqx.field(static=True)
    plane_face: int = eqx.field(static=True)
    curved_face: int = eqx.field(static=True)
    plane_parameter: float = eqx.field(static=True)
    curved_parameter: float = eqx.field(static=True)


@final
class _Seam(StrictModule):
    plane: PlanePatch
    curved: CylinderPatch | ConePatch | SpherePatch
    plane_source: PlanePatch | RevolutionSurface
    curved_source: CylinderPatch | ConePatch | SpherePatch | RevolutionSurface
    plane_profile_interval: tuple[float, float] | None = eqx.field(static=True)
    curved_profile_interval: tuple[float, float] | None = eqx.field(static=True)
    plane_profile_affine: tuple[ExactScalar, ExactScalar] | None = eqx.field(static=True)
    curved_profile_affine: tuple[ExactScalar, ExactScalar] | None = eqx.field(static=True)
    curved_profile_circle: CircleCurve | None
    cone_slope_exact: ExactScalar | None = eqx.field(static=True)
    axis_origin_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = eqx.field(
        static=True
    )
    axis_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = eqx.field(static=True)
    first_axis_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = eqx.field(
        static=True
    )
    second_axis_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = eqx.field(
        static=True
    )
    seam_center_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = eqx.field(
        static=True
    )
    seam_radius_exact: ExactScalar = eqx.field(static=True)
    quadric_radius_exact: ExactScalar = eqx.field(static=True)
    z_exact: ExactScalar = eqx.field(static=True)
    plane_face: int = eqx.field(static=True)
    curved_face: int = eqx.field(static=True)
    plane_coedge: int = eqx.field(static=True)
    curved_coedge: int = eqx.field(static=True)
    seam_edge: int = eqx.field(static=True)
    edge: int = eqx.field(static=True)
    z: float = eqx.field(static=True)
    radial: float = eqx.field(static=True)
    parameter: float = eqx.field(static=True)
    curved_endpoint: int = eqx.field(static=True)
    radial_sense: int = eqx.field(static=True)
    axial_sense: int = eqx.field(static=True)


def _revolution_witness(
    model: BRepModel, face: int, /
) -> RevolutionQuadricWitness | None:
    patch = model.patches[face]
    if not isinstance(patch, RevolutionSurface):
        return None
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    return patch.exact_quadric_witness((float(bounds[0, 1]), float(bounds[1, 1])))


def _face_family(model: BRepModel, face: int, /) -> str | None:
    patch = model.patches[face]
    if isinstance(patch, PlanePatch):
        return "plane"
    if isinstance(patch, CylinderPatch):
        return "cylinder"
    if isinstance(patch, ConePatch):
        return "cone"
    if isinstance(patch, SpherePatch):
        return "sphere"
    witness = _revolution_witness(model, face)
    return None if witness is None else witness.family


def _witness_carrier(
    witness: RevolutionQuadricWitness, /
) -> PlanePatch | CylinderPatch | ConePatch | SpherePatch:
    axis = np.asarray(tuple(float(value) for value in witness.axis_exact))
    first_axis = np.asarray(tuple(float(value) for value in witness.radial_axis_exact))
    second_axis = np.asarray(
        tuple(float(value) for value in witness.second_radial_axis_exact)
    )
    if witness.family == "sphere":
        if witness.center_exact is None or witness.radius_exact is None:
            raise BRepBlendUnsupported(
                "closed seam witness lacks exact sphere coefficients"
            )
        center = np.asarray(tuple(float(value) for value in witness.center_exact))
        return SpherePatch(
            center, first_axis, second_axis, axis, float(witness.radius_exact)
        )
    if witness.origin_exact is None or witness.axial_origin_exact is None:
        raise BRepBlendUnsupported("closed seam witness lacks an axial source origin")
    origin_exact = tuple(
        witness.origin_exact[index]
        + witness.axial_origin_exact * witness.axis_exact[index]
        for index in range(3)
    )
    origin = np.asarray(tuple(float(value) for value in origin_exact))
    if witness.family == "plane":
        return PlanePatch(origin, first_axis, second_axis)
    if witness.family == "cylinder":
        if witness.radius_exact is None:
            raise BRepBlendUnsupported("closed seam witness lacks exact cylinder radius")
        return CylinderPatch(
            origin,
            first_axis,
            second_axis,
            axis,
            float(witness.radius_exact),
            origin_exact=origin_exact,
            first_axis_exact=witness.radial_axis_exact,
            second_axis_exact=witness.second_radial_axis_exact,
            axis_exact=witness.axis_exact,
            radius_exact=witness.radius_exact,
        )
    if witness.family == "cone":
        if witness.radial_origin_exact is None or witness.dr_dz_exact is None:
            raise BRepBlendUnsupported(
                "closed seam witness lacks exact cone coefficients"
            )
        return ConePatch(
            origin,
            first_axis,
            second_axis,
            axis,
            float(witness.radial_origin_exact),
            float(np.arctan(float(witness.dr_dz_exact))),
        )
    raise BRepBlendUnsupported("closed seam witness has an unsupported analytic family")


def _source_axis_point(
    source: PlanePatch | CylinderPatch | ConePatch | SpherePatch | RevolutionSurface,
    witness: RevolutionQuadricWitness | None,
    /,
) -> tuple[ExactScalar, ...] | None:
    if witness is not None:
        if witness.family == "sphere":
            return witness.center_exact
        if witness.origin_exact is None or witness.axial_origin_exact is None:
            return None
        return tuple(
            witness.origin_exact[index]
            + witness.axial_origin_exact * witness.axis_exact[index]
            for index in range(3)
        )
    if isinstance(source, SpherePatch):
        return tuple(rational_vector(np.asarray(source.center)))
    if isinstance(source, (PlanePatch, CylinderPatch, ConePatch)):
        return tuple(rational_vector(np.asarray(source.origin)))
    return None


def _exact_dot3(
    first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
) -> ExactScalar:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)),
        Fraction(0),
    )


def _exact_scalar_bounds(value: ExactScalar, /) -> tuple[float, float]:
    lower, upper = coefficient_interval(value, bits=128)
    return fraction_down(lower), fraction_up(upper)


def _revolution_profile_chart_root(
    ring_pcurve: AbstractCurve, parameter_bounds: tuple[float, float], /
) -> TrimIntersectionRoot:
    second = LineCurve(
        (0.0, 0.0),
        (0.0, 1.0),
        origin_exact=(Fraction(0), Fraction(0)),
        direction_exact=(Fraction(0), Fraction(1)),
    )
    radius = Fraction(1, 8)
    return TrimIntersectionRoot(
        ring_pcurve,
        second,
        parameter_lower=np.asarray(
            (
                -float(radius),
                fraction_down(Fraction.from_float(parameter_bounds[0]) - radius),
            ),
            dtype=np.float64,
        ),
        parameter_upper=np.asarray(
            (
                float(radius),
                fraction_up(Fraction.from_float(parameter_bounds[1]) + radius),
            ),
            dtype=np.float64,
        ),
    )


def _ruled_chart_endpoint_root(
    ring_pcurve: LineCurve,
    generator_pcurve: LineCurve,
    ring_parameter: float,
    generator_parameter: float,
    /,
) -> TrimIntersectionRoot:
    ring_parameter_exact = Fraction.from_float(float(ring_parameter))
    generator_parameter_exact = Fraction.from_float(float(generator_parameter))
    point_exact = tuple(
        origin + ring_parameter_exact * direction
        for origin, direction in zip(
            ring_pcurve.origin_exact, ring_pcurve.direction_exact, strict=True
        )
    )
    generator_point_exact = tuple(
        origin + generator_parameter_exact * direction
        for origin, direction in zip(
            generator_pcurve.origin_exact,
            generator_pcurve.direction_exact,
            strict=True,
        )
    )
    if point_exact != generator_point_exact:
        raise BRepBlendUnsupported(
            "closed seam blend: ruled boundary curves lack an exact shared chart point"
        )
    if point_exact[1] == 0:
        first_direction = (Fraction(1), Fraction(1))
    elif point_exact[1] == 1:
        first_direction = (Fraction(1), Fraction(-1))
    else:
        raise BRepBlendUnsupported(
            "closed seam blend: ruled boundary root is not on a rail endpoint"
        )
    first = LineCurve(
        np.asarray(tuple(float(value) for value in point_exact)),
        np.asarray(tuple(float(value) for value in first_direction)),
        origin_exact=point_exact,
        direction_exact=first_direction,
    )
    isolation_radius = Fraction(1, 8)
    return TrimIntersectionRoot(
        first,
        generator_pcurve,
        parameter_lower=np.asarray(
            (-float(isolation_radius), generator_parameter - float(isolation_radius)),
            dtype=np.float64,
        ),
        parameter_upper=np.asarray(
            (float(isolation_radius), generator_parameter + float(isolation_radius)),
            dtype=np.float64,
        ),
    )


def _mapped_profile_parameter(
    affine: tuple[ExactScalar, ExactScalar] | None,
    parameter: Array | float,
    /,
) -> Array:
    value = jnp.asarray(parameter, dtype=jnp.float64)
    if affine is None:
        return value
    return float(affine[0]) * value + float(affine[1])


def _canonical_profile_parameter(
    affine: tuple[ExactScalar, ExactScalar] | None,
    parameter: Array | float,
    /,
) -> Array:
    value = jnp.asarray(parameter, dtype=jnp.float64)
    if affine is None:
        return value
    return (value - float(affine[1])) / float(affine[0])


@final
class _Meridian(StrictModule):
    patch: TorusPatch | RevolutionSurface | RuledSurface
    plane_radius: float = eqx.field(static=True)
    plane_radius_exact: ExactScalar | None = eqx.field(static=True)
    curved_radius: float = eqx.field(static=True)
    curved_radius_exact: ExactScalar | None = eqx.field(static=True)
    curved_z: float = eqx.field(static=True)
    curved_z_exact: ExactScalar | None = eqx.field(static=True)
    curved_parameter: float = eqx.field(static=True)
    plane_parameter: float = eqx.field(static=True)
    blend_parameter: float = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    contact_bounds: tuple[float, float] = eqx.field(static=True)


def _isoline(
    pcurve: BRepCurve | PeriodicPCurve, /
) -> tuple[ExactScalar, ExactScalar] | None:
    source = pcurve.source_curve if isinstance(pcurve, PeriodicPCurve) else pcurve
    if not isinstance(source, AbstractCurve):
        return None
    coefficients = _affine_curve_coefficients(source, (Fraction(0), Fraction(0)))
    if coefficients is None or coefficients[1] != (Fraction(1), Fraction(0)):
        return None
    return coefficients[0][0], coefficients[0][1]


def _isoparametric_pcurve(curve: SurfaceIsoparametricCurve, /) -> LineCurve:
    fixed = (
        curve.fixed_value_exact
        if curve.fixed_value_exact is not None
        else Fraction.from_float(curve.fixed_value)
    )
    if curve.fixed_axis == 0:
        return LineCurve(
            (curve.fixed_value, 0.0),
            (0.0, 1.0),
            origin_exact=(fixed, Fraction(0)),
            direction_exact=(Fraction(0), Fraction(1)),
        )
    return LineCurve(
        (0.0, curve.fixed_value),
        (1.0, 0.0),
        origin_exact=(Fraction(0), fixed),
        direction_exact=(Fraction(1), Fraction(0)),
    )


def _require_shared_frame(
    plane: PlanePatch, curved: CylinderPatch | ConePatch | SpherePatch
) -> None:
    if (
        exact_frame_orientation(
            np.asarray(curved.first_axis),
            np.asarray(curved.second_axis),
            np.asarray(curved.axis),
        )
        != 1
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: the revolution frame is not exactly right-handed orthonormal"
        )
    if not (
        np.array_equal(plane.first_axis, curved.first_axis)
        and np.array_equal(plane.second_axis, curved.second_axis)
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: plane chart must share the certified circular frame"
        )


def _require_native_turn(geometry: BRepGeometry, edge: int) -> None:
    first, last = geometry.edge_endpoint_roots[edge]
    if not (
        isinstance(first, NativePeriodEndpoint)
        and isinstance(last, NativePeriodEndpoint)
        and first.rational == last.rational == 0
        and last.turns - first.turns == 1
        and first.turns == 0
        and geometry.edge_vertices[edge][0] == geometry.edge_vertices[edge][1]
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: selected edge is not one authored circular native turn"
        )
    carrier_index = geometry.edge_curves[edge]
    if carrier_index < 0 or not isinstance(
        geometry.curves[carrier_index], (CircleCurve, SurfaceIsoparametricCurve)
    ):
        raise BRepBlendUnsupported("closed seam blend: selected carrier is not circular")


def _require_rectangular_chart(
    geometry: BRepGeometry,
    plane_face: int,
    curved_face: int,
    curved_coedge: int,
    source_region: SurfaceRegion,
) -> tuple[float, int, int]:
    pcurve = geometry.pcurves[curved_coedge]
    offset = _isoline(pcurve)
    if offset is not None and offset[0] == 0:
        parameter = float(offset[1])
        parameter_lower, parameter_upper = _exact_scalar_bounds(offset[1])
    elif isinstance(source_region.patch, RevolutionSurface):
        from ._correspondence import _rooted_profile_chart

        edge = geometry.coedge_edges[curved_coedge]
        first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
        root = _rooted_profile_chart(
            pcurve,
            source_region.patch,
            source_region.patch.curve,
            source_region,
            first,
            last,
        )
        if root is None:
            raise BRepBlendUnsupported(
                "closed seam blend: curved boundary lacks an exact circular isoline"
            )
        parameter = root.parameter
        parameter_lower, parameter_upper = root.parameter_lower, root.parameter_upper
    else:
        raise BRepBlendUnsupported(
            "closed seam blend: curved boundary is not a native circular isoline"
        )
    box = np.asarray(source_region.parameter_box)
    lower_endpoint = parameter_lower == box[0, 1]
    upper_endpoint = parameter_upper == box[1, 1]
    if lower_endpoint == upper_endpoint or box[0, 0] != 0:
        raise BRepBlendUnsupported(
            "closed seam blend: circular source is not a unique native chart endpoint"
        )
    endpoint = 0 if lower_endpoint else 1
    loops = geometry.face_loops[curved_face]
    if len(loops) != 1 or len(loops[0]) != 4:
        raise BRepBlendUnsupported(
            "closed seam blend: several-edge curved trims are unsupported"
        )
    counts: dict[int, int] = {}
    for coedge in loops[0]:
        item = geometry.coedge_edges[coedge]
        counts[item] = counts.get(item, 0) + 1
    seams = [item for item, count in counts.items() if count == 2]
    if len(seams) != 1:
        raise BRepBlendUnsupported(
            "closed seam blend: curved face must have one meridian seam"
        )
    plane_loops = geometry.face_loops[plane_face]
    if any(len(loop) != 1 for loop in plane_loops):
        plane_counts: dict[int, int] = {}
        if len(plane_loops) == 1 and len(plane_loops[0]) == 4:
            for coedge in plane_loops[0]:
                item = geometry.coedge_edges[coedge]
                plane_counts[item] = plane_counts.get(item, 0) + 1
        selected_edge = geometry.coedge_edges[curved_coedge]
        if (
            len(plane_loops) != 1
            or len(plane_loops[0]) != 4
            or len(plane_counts) != 3
            or sorted(plane_counts.values()) != [1, 1, 2]
            or plane_counts.get(selected_edge) != 1
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: noncircular or several-edge plane loops are unsupported"
            )
    return parameter, seams[0], endpoint


def _seam(model: BRepModel, edge: int, /) -> _Seam:
    geometry = model.geometry
    if not 0 <= edge < len(geometry.edge_curves):
        raise ValueError(
            f"closed_seam_blend.edge_index must reference the source topology; got {edge}"
        )
    assembly = geometry.assembly_graph
    if (
        geometry.solid_shells != ((0,),)
        or len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != set(range(len(model.patches)))
        or geometry.shell_orientations != ((1,) * len(model.patches),)
        or assembly.solid_count != 1
        or assembly.definitions
        or len(assembly.roots) != 1
        or assembly.roots[0].solid_index != 0
        or assembly.roots[0].placement != BRepAffinePlacement.identity()
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: multiple solids or placed occurrences require materialization"
        )
    uses = [
        (face, coedge)
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
        if geometry.coedge_edges[coedge] == edge
    ]
    if len(uses) != 2 or uses[0][0] == uses[1][0]:
        raise BRepBlendUnsupported(
            "closed seam blend: exactly two distinct incident faces are required"
        )
    plane_uses = [
        (face, coedge) for face, coedge in uses if _face_family(model, face) == "plane"
    ]
    if len(plane_uses) != 1:
        raise BRepBlendUnsupported("closed seam blend: one analytic plane is required")
    plane_face, plane_coedge = plane_uses[0]
    curved_face, curved_coedge = next(use for use in uses if use != plane_uses[0])
    plane_source = model.patches[plane_face]
    curved_source = model.patches[curved_face]
    plane_witness = _revolution_witness(model, plane_face)
    curved_witness = _revolution_witness(model, curved_face)
    curved_family = _face_family(model, curved_face)
    if curved_family not in ("cylinder", "cone", "sphere"):
        raise BRepBlendUnsupported(
            "closed seam blend: curved face is not a cylinder, cone or sphere"
        )
    if isinstance(plane_source, PlanePatch):
        plane = plane_source
    elif isinstance(plane_source, RevolutionSurface) and plane_witness is not None:
        plane_carrier = _witness_carrier(plane_witness)
        if not isinstance(plane_carrier, PlanePatch):
            raise BRepBlendUnsupported(
                "closed seam blend: source cap witness is not planar"
            )
        plane = plane_carrier
    else:
        raise BRepBlendUnsupported(
            "closed seam blend: exact source plane witness is unavailable"
        )
    if isinstance(curved_source, (CylinderPatch, ConePatch, SpherePatch)):
        curved = curved_source
    elif isinstance(curved_source, RevolutionSurface) and curved_witness is not None:
        curved_carrier = _witness_carrier(curved_witness)
        if not isinstance(curved_carrier, (CylinderPatch, ConePatch, SpherePatch)):
            raise BRepBlendUnsupported(
                "closed seam blend: source witness is not a curved quadric"
            )
        curved = curved_carrier
    else:
        raise BRepBlendUnsupported(
            "closed seam blend: exact source quadric witness is unavailable"
        )
    _require_shared_frame(plane, curved)
    _require_native_turn(geometry, edge)
    source_seam_curve = geometry.curves[geometry.edge_curves[edge]]
    if not isinstance(source_seam_curve, CircleCurve):
        raise BRepBlendUnsupported(
            "closed seam blend: exact circular source carrier is unavailable"
        )
    parameter, seam_edge, curved_endpoint = _require_rectangular_chart(
        geometry,
        plane_face,
        curved_face,
        curved_coedge,
        SurfaceRegion(curved_source, np.asarray(model.parameter_bounds[curved_face])),
    )
    plane_origin = _source_axis_point(plane_source, plane_witness)
    curved_origin = _source_axis_point(curved_source, curved_witness)
    axis_exact = (
        curved_witness.axis_exact
        if curved_witness is not None
        else tuple(rational_vector(np.asarray(curved.axis)))
    )
    if plane_origin is None or curved_origin is None:
        raise BRepBlendUnsupported(
            "closed seam blend: exact revolution axis origin is unavailable"
        )
    relative_exact = tuple(
        plane_origin[index] - curved_origin[index] for index in range(3)
    )
    axis_squared = _exact_dot3(axis_exact, axis_exact)
    if axis_squared == 0:
        raise BRepBlendUnsupported("closed seam blend: revolution axis is singular")
    height_exact = _exact_dot3(relative_exact, axis_exact) / axis_squared
    if relative_exact != tuple(height_exact * value for value in axis_exact):
        raise BRepBlendUnsupported(
            "closed seam blend: plane origin is not on the certified revolution axis"
        )
    center = np.asarray(tuple(float(value) for value in curved_origin))
    point = np.asarray(
        curved_source.evaluate(jnp.asarray((0.0, parameter), dtype=jnp.float64)),
        dtype=np.float64,
    )
    radial = float((point - center) @ np.asarray(curved.first_axis))
    plane_interval = None
    if plane_witness is not None:
        witness_interval = plane_witness.profile_parameter_interval
        if len(witness_interval) != 2:
            raise BRepBlendUnsupported(
                "closed seam blend: planar revolution profile interval is unavailable"
            )
        plane_interval = (float(witness_interval[0]), float(witness_interval[1]))
    curved_interval = None
    if curved_witness is not None:
        witness_interval = curved_witness.profile_parameter_interval
        if len(witness_interval) != 2:
            raise BRepBlendUnsupported(
                "closed seam blend: curved revolution profile interval is unavailable"
            )
        curved_interval = (float(witness_interval[0]), float(witness_interval[1]))
    plane_profile_affine: tuple[ExactScalar, ExactScalar] | None = None
    if plane_witness is not None:
        if (
            plane_witness.radial_origin_exact is None
            or plane_witness.radial_direction_exact in (None, 0)
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: source cap lacks an invertible exact radial profile"
            )
        plane_profile_affine = (
            Fraction(1) / plane_witness.radial_direction_exact,
            -plane_witness.radial_origin_exact / plane_witness.radial_direction_exact,
        )
    curved_profile_affine: tuple[ExactScalar, ExactScalar] | None = None
    curved_profile_circle: CircleCurve | None = None
    cone_slope_exact = None
    if curved_witness is not None:
        if curved_family in ("cylinder", "cone"):
            if curved_witness.axial_direction_exact in (None, 0):
                raise BRepBlendUnsupported(
                    "closed seam blend: curved profile lacks an invertible exact axial map"
                )
            curved_profile_affine = (
                Fraction(1) / curved_witness.axial_direction_exact,
                Fraction(0),
            )
        elif curved_family == "sphere":
            profile = curved_witness.source.curve
            if not isinstance(profile, CircleCurve) or not (
                _exact_dot3(profile.first_axis_exact, curved_witness.radial_axis_exact)
                == 1
                and _exact_dot3(profile.second_axis_exact, curved_witness.axis_exact) == 1
            ):
                raise BRepBlendUnsupported(
                    "closed seam blend: spherical profile latitude is not an exact source chart"
                )
            curved_profile_affine = (Fraction(1), Fraction(0))
            curved_profile_circle = profile
        cone_slope_exact = curved_witness.dr_dz_exact
    if isinstance(curved, SpherePatch) and curved_profile_circle is None:
        sphere_center_exact = tuple(rational_vector(np.asarray(curved.center)))
        sphere_radial_exact = tuple(rational_vector(np.asarray(curved.first_axis)))
        sphere_axis_exact = tuple(rational_vector(np.asarray(curved.axis)))
        if not (
            _exact_dot3(sphere_radial_exact, sphere_radial_exact) == 1
            and _exact_dot3(sphere_axis_exact, sphere_axis_exact) == 1
            and _exact_dot3(sphere_radial_exact, sphere_axis_exact) == 0
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: sphere lacks an exact unit meridian frame"
            )
        curved_profile_circle = CircleCurve(
            np.asarray(curved.center),
            np.asarray(curved.first_axis),
            np.asarray(curved.axis),
            float(curved.radius),
            center_exact=sphere_center_exact,
            first_axis_exact=sphere_radial_exact,
            second_axis_exact=sphere_axis_exact,
            radius_exact=Fraction.from_float(float(curved.radius)),
        )
    first_axis_exact_3 = (
        source_seam_curve.first_axis_exact[0],
        source_seam_curve.first_axis_exact[1],
        source_seam_curve.first_axis_exact[2],
    )
    second_axis_exact_3 = (
        source_seam_curve.second_axis_exact[0],
        source_seam_curve.second_axis_exact[1],
        source_seam_curve.second_axis_exact[2],
    )
    axis_exact_3 = (
        first_axis_exact_3[1] * second_axis_exact_3[2]
        - first_axis_exact_3[2] * second_axis_exact_3[1],
        first_axis_exact_3[2] * second_axis_exact_3[0]
        - first_axis_exact_3[0] * second_axis_exact_3[2],
        first_axis_exact_3[0] * second_axis_exact_3[1]
        - first_axis_exact_3[1] * second_axis_exact_3[0],
    )
    if axis_exact_3 != (axis_exact[0], axis_exact[1], axis_exact[2]):
        raise BRepBlendUnsupported(
            "closed seam blend: source circle frame does not match the exact revolution axis"
        )
    axis_origin_exact = (curved_origin[0], curved_origin[1], curved_origin[2])
    seam_center_exact = (
        source_seam_curve.center_exact[0],
        source_seam_curve.center_exact[1],
        source_seam_curve.center_exact[2],
    )
    expected_center_exact = tuple(
        axis_origin_exact[index] + height_exact * axis_exact_3[index]
        for index in range(3)
    )
    if seam_center_exact != expected_center_exact:
        raise BRepBlendUnsupported(
            "closed seam blend: source circle center lacks an exact revolution axis lift"
        )
    quadric_radius_exact = (
        curved_witness.radius_exact
        if curved_witness is not None and curved_witness.radius_exact is not None
        else Fraction.from_float(float(curved.radius))
        if isinstance(curved, SpherePatch)
        else source_seam_curve.radius_exact
    )
    seam_radius_exact = source_seam_curve.radius_exact
    return _Seam(
        plane=plane,
        curved=curved,
        plane_source=plane_source,
        curved_source=curved_source,
        plane_profile_interval=plane_interval,
        curved_profile_interval=curved_interval,
        plane_profile_affine=plane_profile_affine,
        curved_profile_affine=curved_profile_affine,
        curved_profile_circle=curved_profile_circle,
        cone_slope_exact=cone_slope_exact,
        axis_origin_exact=axis_origin_exact,
        axis_exact=axis_exact_3,
        first_axis_exact=first_axis_exact_3,
        second_axis_exact=second_axis_exact_3,
        seam_center_exact=seam_center_exact,
        seam_radius_exact=seam_radius_exact,
        quadric_radius_exact=quadric_radius_exact,
        z_exact=height_exact,
        plane_face=plane_face,
        curved_face=curved_face,
        plane_coedge=plane_coedge,
        curved_coedge=curved_coedge,
        seam_edge=seam_edge,
        edge=edge,
        z=float(height_exact),
        radial=radial,
        parameter=parameter,
        curved_endpoint=curved_endpoint,
        radial_sense=int(model.orientation[curved_face]),
        axial_sense=int(model.orientation[plane_face]),
    )


def _cone_slope(seam: _Seam, /) -> Array:
    if seam.cone_slope_exact is not None:
        return jnp.asarray(float(seam.cone_slope_exact), dtype=jnp.float64)
    if not isinstance(seam.curved, ConePatch):
        raise RuntimeError("A cone slope requires a cone carrier.")
    return jnp.tan(seam.curved.semi_angle)


def _cone_slope_bounds(seam: _Seam, /) -> tuple[float, float] | None:
    if seam.cone_slope_exact is None:
        return None
    lower, upper = coefficient_interval(seam.cone_slope_exact, bits=128)
    return float(fraction_down(lower)), float(fraction_up(upper))


def _cone_slope_value(seam: _Seam, /) -> float:
    slope = seam.cone_slope_exact
    if slope is None:
        raise RuntimeError("An exact Revolution cone slope is unavailable.")
    return float(slope)


def _fillet_values(seam: _Seam, radius: Array, slope: Array | None = None, /) -> Array:
    """(center radial, center axial, contact radial, axial, chart parameter)."""
    q, d, z = seam.radial_sense, seam.axial_sense, seam.z
    zc = z - d * radius
    match seam.curved:
        case CylinderPatch():
            rc = seam.curved.radius - q * radius
            return jnp.stack((rc, zc, seam.curved.radius, zc, zc))
        case ConePatch():
            slope = _cone_slope(seam) if slope is None else slope
            length = jnp.sqrt(1 + slope * slope)
            rc = seam.curved.reference_radius + slope * zc - q * radius * length
            zt = zc - q * radius * slope / length
            rt = rc + q * radius / length
            return jnp.stack((rc, zc, rt, zt, zt))
        case SpherePatch():
            inner = seam.curved.radius - q * radius
            rc = jnp.sqrt(inner * inner - zc * zc)
            rt = seam.curved.radius * rc / inner
            zt = seam.curved.radius * zc / inner
            return jnp.stack((rc, zc, rt, zt, zt))


def _fillet(seam: _Seam, radius: float, /) -> _Meridian:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            f"closed_seam_blend.radius must be finite and positive; got {radius}"
        )

    slope_bounds = _cone_slope_bounds(seam)

    def values(point: Array) -> Array:
        slope = None if slope_bounds is None else point[1]
        return _fillet_values(seam, point[0], slope)

    prepared = prepare_interval_function(
        values, 1 if slope_bounds is None else 2, batch_capacity=1
    )
    if slope_bounds is None:
        input_low = input_high = input_host = np.asarray(((radius,),), dtype=np.float64)
    else:
        slope_low, slope_high = slope_bounds
        input_low = np.asarray(((radius, slope_low),), dtype=np.float64)
        input_high = np.asarray(((radius, slope_high),), dtype=np.float64)
        input_host = np.asarray(((radius, _cone_slope_value(seam)),), dtype=np.float64)
    low, high = prepared.evaluate(input_low, input_high)
    if (
        not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or low[0, 0] <= 0
        or low[0, 2] <= 0
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: rolling-ball center is uncertified"
        )
    rc, zc, rt, zt, parameter = (
        float(value) for value in prepared.evaluate_host_points(input_host)[0]
    )
    if isinstance(seam.curved, SpherePatch):
        parameter = atan2(zt, rt)
    radius_exact = Fraction.from_float(radius)
    plane_radius_exact: ExactScalar | None = None
    curved_radius_exact: ExactScalar | None = None
    curved_z_exact: ExactScalar | None = None
    center_z_exact = seam.z_exact - seam.axial_sense * radius_exact
    if isinstance(seam.curved, CylinderPatch):
        plane_radius_exact = seam.seam_radius_exact - seam.radial_sense * radius_exact
        curved_radius_exact = seam.seam_radius_exact
        curved_z_exact = center_z_exact
    elif isinstance(seam.curved, ConePatch) and seam.cone_slope_exact is not None:
        slope_exact = seam.cone_slope_exact
        length_squared_exact = Fraction(1) + slope_exact * slope_exact
        length_squared_fraction: Fraction | None
        try:
            length_squared_fraction = rational_coefficient(
                length_squared_exact, role="closed-seam cone length squared"
            )
        except (TypeError, ValueError):
            length_squared_fraction = None
        if length_squared_fraction is not None and length_squared_fraction > 0:
            length_exact = RadicalElement.sqrt(length_squared_fraction)
            plane_radius_exact = (
                seam.seam_radius_exact
                - seam.axial_sense * slope_exact * radius_exact
                - seam.radial_sense * radius_exact * length_exact
            )
            curved_z_exact = (
                center_z_exact
                - seam.radial_sense * radius_exact * slope_exact / length_exact
            )
            curved_radius_exact = (
                plane_radius_exact + seam.radial_sense * radius_exact / length_exact
            )
    elif isinstance(seam.curved, SpherePatch):
        inner_exact = seam.quadric_radius_exact - seam.radial_sense * radius_exact
        radius_squared_exact = inner_exact * inner_exact - center_z_exact * center_z_exact
        if isinstance(radius_squared_exact, Fraction) and radius_squared_exact > 0:
            plane_radius_exact = RadicalElement.sqrt(radius_squared_exact)
            curved_radius_exact = (
                seam.quadric_radius_exact * plane_radius_exact / inner_exact
            )
            curved_z_exact = seam.quadric_radius_exact * center_z_exact / inner_exact
    if (
        plane_radius_exact is None
        or curved_radius_exact is None
        or curved_z_exact is None
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: exact meridian contact is unavailable"
        )
    rc = float(plane_radius_exact)
    rt = float(curved_radius_exact)
    zt = float(curved_z_exact)
    zc = float(center_z_exact)
    if isinstance(seam.curved, SpherePatch):
        parameter = atan2(zt, rt)
    contact_bounds = (float(low[0, 4]), float(high[0, 4]))
    if isinstance(seam.curved, SpherePatch):
        if seam.curved_profile_circle is None:
            raise BRepBlendUnsupported(
                "closed seam blend: exact sphere latitude root is unavailable"
            )
        contact_point_exact = tuple(
            seam.axis_origin_exact[index]
            + curved_z_exact * seam.axis_exact[index]
            + curved_radius_exact * seam.first_axis_exact[index]
            for index in range(3)
        )
        profile_root = ConicParameterRoot.from_exact_point(
            seam.curved_profile_circle, contact_point_exact, parameter
        )
        if profile_root is None:
            raise BRepBlendUnsupported(
                "closed seam blend: exact sphere latitude root could not be isolated"
            )
        contact_bounds = (profile_root.parameter_lower, profile_root.parameter_upper)
    center_exact = tuple(
        seam.axis_origin_exact[index] + center_z_exact * seam.axis_exact[index]
        for index in range(3)
    )
    patch = TorusPatch(
        np.asarray(tuple(float(value) for value in center_exact), dtype=np.float64),
        np.asarray(
            tuple(float(value) for value in seam.first_axis_exact), dtype=np.float64
        ),
        np.asarray(
            tuple(float(value) for value in seam.second_axis_exact), dtype=np.float64
        ),
        np.asarray(tuple(float(value) for value in seam.axis_exact), dtype=np.float64),
        rc,
        radius,
        center_exact=center_exact,
        first_axis_exact=seam.first_axis_exact,
        second_axis_exact=seam.second_axis_exact,
        axis_exact=seam.axis_exact,
        major_radius_exact=plane_radius_exact,
        minor_radius_exact=radius_exact,
    )
    plane_angle = seam.axial_sense * pi / 2
    curved_angle = atan2(zt - zc, rt - rc)
    cosine_exact = (curved_radius_exact - plane_radius_exact) / radius_exact
    sine_exact = (curved_z_exact - center_z_exact) / radius_exact
    if (
        rational_coefficient(
            cosine_exact * cosine_exact + sine_exact * sine_exact,
            role="closed-seam torus contact direction",
        )
        != 1
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: torus contact is not a unit direction"
        )
    principal_angle = atan2(float(sine_exact), float(cosine_exact))
    if principal_angle < 0.0:
        principal_angle += 2 * pi
    turns = int(round((curved_angle - principal_angle) / (2 * pi)))
    curved_angle = principal_angle + turns * 2 * pi
    while curved_angle - plane_angle > pi:
        curved_angle -= 2 * pi
    while plane_angle - curved_angle > pi:
        curved_angle += 2 * pi
    # Preserve the chosen short arc while placing its start in the native
    # nonnegative gauge. The chart window may cross the periodic seam.
    phase_shift = 2 * pi if min(plane_angle, curved_angle) < 0 else 0.0
    plane_angle += phase_shift
    curved_angle += phase_shift

    from .._interval_enclosure import interval_cos
    from ._patches import _conic_parameter_root_bounds

    plane_turns = int(bool(phase_shift)) - int(seam.axial_sense < 0)
    curved_turns = int(round((curved_angle - principal_angle) / (2 * pi)))
    plane_interval = _conic_parameter_root_bounds(
        Fraction(0), Fraction(seam.axial_sense), plane_turns
    )
    curved_interval = _conic_parameter_root_bounds(cosine_exact, sine_exact, curved_turns)
    cosine_interval = interval_cos(
        (
            np.asarray(min(plane_interval[0], curved_interval[0]), dtype=np.float64),
            np.asarray(max(plane_interval[1], curved_interval[1]), dtype=np.float64),
        )
    )
    cosine_bounds = tuple(Fraction.from_float(float(value)) for value in cosine_interval)
    major_lower, _ = coefficient_interval(patch.major_radius_exact)
    minor_bounds = coefficient_interval(patch.minor_radius_exact)
    radial_lower = major_lower + min(
        minor * cosine for minor in minor_bounds for cosine in cosine_bounds
    )
    if radial_lower <= 0:
        raise BRepBlendUnsupported(
            "closed seam blend: source arc is not certified clear of the revolution axis"
        )
    return _Meridian(
        patch=patch,
        plane_radius=rc,
        plane_radius_exact=plane_radius_exact,
        curved_radius=rt,
        curved_radius_exact=curved_radius_exact,
        curved_z=zt,
        curved_z_exact=curved_z_exact,
        curved_parameter=parameter,
        plane_parameter=plane_angle,
        blend_parameter=curved_angle,
        lower=min(plane_angle, curved_angle),
        upper=max(plane_angle, curved_angle),
        contact_bounds=contact_bounds,
    )


def _chamfer(seam: _Seam, setbacks: tuple[float, float], /) -> _Meridian:
    a, b = setbacks
    if any(not np.isfinite(value) or value <= 0 for value in setbacks):
        raise ValueError(
            f"closed_seam_blend.setbacks must be finite and positive; got {setbacks}"
        )
    q, d = seam.radial_sense, seam.axial_sense

    slope_bounds = _cone_slope_bounds(seam)
    sphere_pcurve: AffinePCurve | None = None
    if (
        isinstance(seam.curved, SpherePatch)
        and isinstance(seam.curved_source, RevolutionSurface)
        and seam.curved_profile_circle is not None
    ):
        seam_point = (
            seam.seam_center_exact[0] + seam.seam_radius_exact * seam.first_axis_exact[0],
            seam.seam_center_exact[1] + seam.seam_radius_exact * seam.first_axis_exact[1],
            seam.seam_center_exact[2] + seam.seam_radius_exact * seam.first_axis_exact[2],
        )
        seam_root = ConicParameterRoot.from_exact_point(
            seam.curved_profile_circle, seam_point, seam.parameter
        )
        if seam_root is None:
            raise BRepBlendUnsupported(
                "closed seam chamfer: exact sphere seam parameter is unavailable"
            )
        sphere_pcurve = AffinePCurve(
            ConicRootLinePCurve(
                seam_root,
                Fraction(0),
                Fraction(1),
                phase_shift_exact=-d * Fraction.from_float(b) / seam.quadric_radius_exact,
                parameter_range=None,
            ),
            ((0, 1), (1, 0)),
            (0, 0),
        )
    rooted_sphere = sphere_pcurve is not None

    def contacts(point: Array) -> Array:
        rp = seam.radial - q * point[0]
        match seam.curved:
            case CylinderPatch():
                parameter = zt = seam.z - d * point[1]
                rt = seam.curved.radius
            case ConePatch():
                slope = _cone_slope(seam) if slope_bounds is None else point[2]
                parameter = zt = seam.z - d * point[1] / jnp.sqrt(1 + slope * slope)
                rt = seam.curved.reference_radius + slope * zt
            case SpherePatch():
                parameter = (
                    point[2]
                    if rooted_sphere
                    else seam.parameter - d * point[1] / seam.curved.radius
                )
                zt = seam.curved.radius * jnp.sin(parameter)
                rt = seam.curved.radius * jnp.cos(parameter)
        return jnp.stack((rp, rt, zt, parameter))

    plan = prepare_interval_function(
        contacts,
        3 if rooted_sphere or slope_bounds is not None else 2,
        batch_capacity=1,
    )
    if sphere_pcurve is not None:
        sphere_bounds = sphere_pcurve.bounding_box(0.0, 0.0)
        sphere_parameter = float(np.asarray(sphere_pcurve.evaluate(jnp.asarray(0.0)))[1])
        inputs_low = np.asarray(((a, b, sphere_bounds[0, 1]),), dtype=np.float64)
        inputs_high = np.asarray(((a, b, sphere_bounds[1, 1]),), dtype=np.float64)
        inputs_host = np.asarray(((a, b, sphere_parameter),), dtype=np.float64)
    elif slope_bounds is None:
        inputs_low = inputs_high = inputs_host = np.asarray(((a, b),), dtype=np.float64)
    else:
        slope_low, slope_high = slope_bounds
        inputs_low = np.asarray(((a, b, slope_low),), dtype=np.float64)
        inputs_high = np.asarray(((a, b, slope_high),), dtype=np.float64)
        inputs_host = np.asarray(((a, b, _cone_slope_value(seam)),), dtype=np.float64)
    low, high = plan.evaluate(inputs_low, inputs_high)
    if (
        not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or min(low[0, :2]) <= 0
        or low[0, 2] <= seam.z <= high[0, 2]
    ):
        raise BRepBlendUnsupported(
            "closed seam blend: chamfer axis avoidance or axial separation is uncertified"
        )
    rp, rt, zt, parameter = (
        float(value) for value in plan.evaluate_host_points(inputs_host)[0]
    )
    plane_radius_exact = seam.seam_radius_exact - q * Fraction.from_float(a)
    if float(plane_radius_exact) != rp:
        raise BRepBlendUnsupported(
            "closed seam blend: exact chamfer cap contact does not match represented meridian"
        )
    curved_radius_exact: ExactScalar | None = None
    curved_z_exact: ExactScalar | None = None
    setback_exact = Fraction.from_float(b)
    if isinstance(seam.curved, CylinderPatch):
        curved_radius_exact = seam.curved.radius_exact
        curved_z_exact = seam.z_exact - d * setback_exact
    elif isinstance(seam.curved, ConePatch) and seam.cone_slope_exact is not None:
        slope_exact = seam.cone_slope_exact
        length_squared_exact = Fraction(1) + slope_exact * slope_exact
        try:
            length_squared_fraction = rational_coefficient(
                length_squared_exact, role="closed-seam chamfer cone length squared"
            )
        except (TypeError, ValueError):
            length_squared_fraction = None
        if length_squared_fraction is not None and length_squared_fraction > 0:
            length_exact = RadicalElement.sqrt(length_squared_fraction)
            curved_z_exact = seam.z_exact - d * setback_exact / length_exact
            curved_radius_exact = seam.seam_radius_exact + slope_exact * (
                curved_z_exact - seam.z_exact
            )
    if curved_radius_exact is not None and curved_z_exact is not None:
        exact_radius = float(curved_radius_exact)
        exact_height = float(curved_z_exact)
        if not (
            low[0, 1] <= exact_radius <= high[0, 1]
            and low[0, 2] <= exact_height <= high[0, 2]
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: exact chamfer source contact is outside its interval proof"
            )
        rt, zt, parameter = exact_radius, exact_height, exact_height
    if not (high[0, 0] < low[0, 1] or high[0, 1] < low[0, 0]):
        raise BRepBlendUnsupported(
            "closed seam blend: chamfer generatrix slope sign is uncertified"
        )
    if sphere_pcurve is not None:
        if (
            not isinstance(seam.curved_source, RevolutionSurface)
            or seam.curved_profile_circle is None
        ):
            raise BRepBlendUnsupported(
                "closed seam chamfer: exact source Revolution sphere rail is unavailable"
            )
        source_profile = seam.curved_profile_circle
        if (
            source_profile.center_exact != seam.axis_origin_exact
            or source_profile.first_axis_exact != seam.first_axis_exact
            or source_profile.second_axis_exact != seam.axis_exact
        ):
            raise BRepBlendUnsupported(
                "closed seam chamfer: exact Revolution sphere ring frame is not aligned with the authored seam"
            )
        plane_ring = _circle(
            seam,
            rp,
            plane_radius_exact,
            axial_exact=seam.z_exact,
        )
        sphere_ring = SurfaceMappedCurve(seam.curved_source, sphere_pcurve)
        patch = RuledSurface(plane_ring, sphere_ring)
        last = 1.0
        lower, upper = 0.0, 1.0
    else:
        if curved_radius_exact is None or curved_z_exact is None:
            raise BRepBlendUnsupported(
                "closed seam chamfer: exact generator contacts are unavailable"
            )
        plane_point = tuple(
            seam.axis_origin_exact[index]
            + seam.z_exact * seam.axis_exact[index]
            + plane_radius_exact * seam.first_axis_exact[index]
            for index in range(3)
        )
        contact_point = tuple(
            seam.axis_origin_exact[index]
            + curved_z_exact * seam.axis_exact[index]
            + curved_radius_exact * seam.first_axis_exact[index]
            for index in range(3)
        )
        direction = tuple(
            last - first for first, last in zip(plane_point, contact_point, strict=True)
        )
        patch = RevolutionSurface(
            LineCurve(
                np.asarray(
                    tuple(float(value) for value in plane_point), dtype=np.float64
                ),
                np.asarray(tuple(float(value) for value in direction), dtype=np.float64),
                origin_exact=plane_point,
                direction_exact=direction,
            ),
            np.asarray(
                tuple(float(value) for value in seam.axis_origin_exact), dtype=np.float64
            ),
            np.asarray(
                tuple(float(value) for value in seam.axis_exact), dtype=np.float64
            ),
        )
        last, lower, upper = 1.0, 0.0, 1.0
    return _Meridian(
        patch=patch,
        plane_radius=rp,
        plane_radius_exact=plane_radius_exact,
        curved_radius=rt,
        curved_radius_exact=curved_radius_exact,
        curved_z=zt,
        curved_z_exact=curved_z_exact,
        curved_parameter=parameter,
        plane_parameter=0.0,
        blend_parameter=last,
        lower=lower,
        upper=upper,
        contact_bounds=(float(low[0, 3]), float(high[0, 3])),
    )


def _copy_builder(model: BRepModel, /) -> _Builder:
    geometry = model.geometry
    builder = _Builder()
    vertices = tuple(
        builder.vertex(
            np.asarray(geometry.vertex_points[index]),
            exact_point=geometry.vertex_exact_points[index],
            root=geometry.vertex_roots[index],
        )
        for index in range(len(geometry.vertex_points))
    )
    edge_ranges = np.asarray(geometry.edge_ranges)
    for edge, (start, end) in enumerate(geometry.edge_vertices):
        curve_index = geometry.edge_curves[edge]
        roots = geometry.edge_endpoint_roots[edge]
        if curve_index < 0:
            builder.degenerate(
                vertices[start],
                edge_ranges[edge, 0],
                edge_ranges[edge, 1],
                endpoint_roots=roots,
            )
        else:
            builder.edge(
                geometry.curves[curve_index],
                edge_ranges[edge, 0],
                edge_ranges[edge, 1],
                vertices[start],
                vertices[end],
                endpoint_roots=roots,
            )
    for coedge, edge in enumerate(geometry.coedge_edges):
        builder.coedge(
            edge,
            geometry.coedge_senses[coedge],
            geometry.pcurves[coedge],
            endpoint_roots=geometry.coedge_endpoint_roots[coedge],
        )
    for face, patch in enumerate(model.patches):
        builder.face(
            patch,
            np.asarray(model.parameter_bounds[face]),
            int(model.orientation[face]),
            model.physical_tags[face],
            [list(loop) for loop in geometry.face_loops[face]],
        )
    return builder


def _circle(
    seam: _Seam,
    radius: float,
    radius_exact: ExactScalar | None,
    /,
    *,
    axial_exact: ExactScalar | None = None,
) -> CircleCurve:
    center_exact = (
        seam.seam_center_exact
        if axial_exact is None
        else tuple(
            seam.axis_origin_exact[index] + axial_exact * seam.axis_exact[index]
            for index in range(3)
        )
    )
    return CircleCurve(
        np.asarray(tuple(float(value) for value in center_exact)),
        np.asarray(tuple(float(value) for value in seam.first_axis_exact)),
        np.asarray(tuple(float(value) for value in seam.second_axis_exact)),
        radius,
        center_exact=center_exact,
        first_axis_exact=seam.first_axis_exact,
        second_axis_exact=seam.second_axis_exact,
        radius_exact=radius_exact,
    )


def _require_clearance(model: BRepModel, seam: _Seam, meridian: _Meridian) -> None:
    box = np.asarray(model.parameter_bounds[seam.curved_face])
    source_low, source_high = map(float, box[:, 1])
    mapped = tuple(
        float(np.asarray(_canonical_profile_parameter(seam.curved_profile_affine, value)))
        for value in (source_low, source_high)
    )
    low, high = min(mapped), max(mapped)
    contact_low, contact_high = meridian.contact_bounds
    if not low < contact_low <= contact_high < high:
        raise BRepBlendUnsupported(
            "closed seam blend: blend reaches another boundary or a pole"
        )
    for canonical, affine, interval, label in (
        (
            meridian.plane_radius,
            seam.plane_profile_affine,
            seam.plane_profile_interval,
            "planar cap",
        ),
        (
            meridian.curved_parameter,
            seam.curved_profile_affine,
            seam.curved_profile_interval,
            "curved face",
        ),
    ):
        if interval is None:
            continue
        parameter = float(np.asarray(_mapped_profile_parameter(affine, canonical)))
        profile_low, profile_high = sorted(interval)
        if not profile_low < parameter < profile_high:
            raise BRepBlendUnsupported(
                f"closed seam blend: contact leaves the exact {label} profile interval"
            )

    geometry = model.geometry
    plane_edge_uses: dict[int, int] = {}
    for loop in geometry.face_loops[seam.plane_face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            plane_edge_uses[edge] = plane_edge_uses.get(edge, 0) + 1
    for loop in geometry.face_loops[seam.plane_face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            if edge == seam.edge or plane_edge_uses[edge] == 2:
                continue
            curve_index = geometry.edge_curves[edge]
            if curve_index < 0:
                vertex = geometry.edge_vertices[edge][0]
                exact_point = geometry.vertex_exact_points[vertex]
                if exact_point != seam.seam_center_exact:
                    raise BRepBlendUnsupported(
                        "closed seam blend: a degenerate plane trim is not on its axis"
                    )
                other = 0.0
            elif isinstance(seam.plane_source, RevolutionSurface):
                curve = geometry.curves[curve_index]
                if (
                    not isinstance(curve, CircleCurve)
                    or curve.center_exact != seam.seam_center_exact
                ):
                    raise BRepBlendUnsupported(
                        "closed seam blend: a Revolution cap trim is not an exact concentric circle"
                    )
                other = float(curve.radius_exact)
            else:
                pcurve = geometry.pcurves[coedge]
                if not isinstance(pcurve, CircleCurve) or not np.array_equal(
                    pcurve.center, np.zeros(2)
                ):
                    raise BRepBlendUnsupported(
                        "closed seam blend: nonconcentric plane loops are unsupported"
                    )
                other = float(pcurve.radius)
            if (
                min(seam.radial, meridian.plane_radius)
                <= other
                <= max(seam.radial, meridian.plane_radius)
            ):
                raise BRepBlendUnsupported(
                    "closed seam blend: setback reaches another circular plane trim"
                )


def _stage_boundaries(
    builder: _Builder,
    geometry: BRepGeometry,
    seam: _Seam,
    meridian: _Meridian,
    /,
) -> tuple[int, int]:
    if isinstance(meridian.patch, RuledSurface):
        plane_curve = meridian.patch.first
        if not isinstance(plane_curve, CircleCurve):
            raise BRepBlendUnsupported(
                "closed seam chamfer: ruled plane rail is not a native circle"
            )
    else:
        plane_curve = _circle(seam, meridian.plane_radius, meridian.plane_radius_exact)
    builder.curves[builder.edge_curves[seam.edge]] = plane_curve
    vertex = builder.edge_vertices[seam.edge][0]
    plane_point_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
        plane_curve.center_exact[0]
        + plane_curve.radius_exact * plane_curve.first_axis_exact[0],
        plane_curve.center_exact[1]
        + plane_curve.radius_exact * plane_curve.first_axis_exact[1],
        plane_curve.center_exact[2]
        + plane_curve.radius_exact * plane_curve.first_axis_exact[2],
    )
    builder.vertices[vertex] = np.asarray(
        tuple(float(value) for value in plane_point_exact), dtype=np.float64
    )
    rational_plane_point: tuple[Fraction, Fraction, Fraction] | None = None
    if (
        isinstance(plane_point_exact[0], Fraction)
        and isinstance(plane_point_exact[1], Fraction)
        and isinstance(plane_point_exact[2], Fraction)
    ):
        rational_plane_point = (
            plane_point_exact[0],
            plane_point_exact[1],
            plane_point_exact[2],
        )
    builder.vertex_exact_points[vertex] = rational_plane_point
    plane_vertex_conic = ConicParameterRoot(plane_curve, Fraction(1), Fraction(0))
    if plane_vertex_conic.exact_point() != plane_point_exact:
        raise BRepBlendUnsupported(
            "closed seam blend: plane rail vertex lacks its exact conic point"
        )
    builder.vertex_roots[vertex] = BRepVertexRoot(BRepConicVertex(plane_vertex_conic))
    plane_profile_parameter_exact = (
        None
        if seam.plane_profile_affine is None or meridian.plane_radius_exact is None
        else seam.plane_profile_affine[0] * meridian.plane_radius_exact
        + seam.plane_profile_affine[1]
    )
    plane_profile_parameter = (
        float(plane_profile_parameter_exact)
        if plane_profile_parameter_exact is not None
        else float(
            np.asarray(
                _mapped_profile_parameter(
                    seam.plane_profile_affine, meridian.plane_radius
                )
            )
        )
    )
    if seam.plane_profile_affine is None:
        plane_pcurve: BRepCurve = CircleCurve(
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), meridian.plane_radius
        )
        plane_coedge_roots = (
            NativePeriodEndpoint(plane_pcurve, rational=0, turns=0),
            NativePeriodEndpoint(plane_pcurve, rational=0, turns=1),
        )
    else:
        if not isinstance(seam.plane_source, RevolutionSurface):
            raise RuntimeError("A mapped plane profile lost its revolution source.")
        plane_pcurve = LineCurve(
            (0.0, plane_profile_parameter),
            (1.0, 0.0),
            origin_exact=(
                None
                if plane_profile_parameter_exact is None
                else (Fraction(0), plane_profile_parameter_exact)
            ),
            direction_exact=(Fraction(1), Fraction(0)),
        )
        plane_coedge_roots = (
            NativePeriodEndpoint(plane_pcurve, seam.plane_source, 0, rational=0, turns=0),
            NativePeriodEndpoint(plane_pcurve, seam.plane_source, 0, rational=0, turns=1),
        )
    builder.pcurves[seam.plane_coedge] = plane_pcurve
    plane_roots = (
        NativePeriodEndpoint(plane_curve, rational=0, turns=0),
        NativePeriodEndpoint(plane_curve, rational=0, turns=1),
    )
    builder.edge_endpoint_roots[seam.edge] = plane_roots
    builder.coedge_endpoint_roots[seam.plane_coedge] = plane_coedge_roots
    ruled_plane_root: TrimIntersectionRoot | None = None
    ruled_plane_ring_pcurve: LineCurve | None = None
    ruled_sphere_ring_pcurve: LineCurve | None = None
    ruled_generator_pcurve: LineCurve | None = None
    ruled_sphere_root: TrimIntersectionRoot | None = None
    ruled_sphere_lift: BRepCurveSurfaceLift | None = None
    if isinstance(meridian.patch, RuledSurface):
        ruled_plane_ring_pcurve = LineCurve(
            (0.0, 0.0),
            (1.0, 0.0),
            origin_exact=(Fraction(0), Fraction(0)),
            direction_exact=(Fraction(1), Fraction(0)),
        )
        ruled_sphere_ring_pcurve = LineCurve(
            (0.0, 1.0),
            (1.0, 0.0),
            origin_exact=(Fraction(0), Fraction(1)),
            direction_exact=(Fraction(1), Fraction(0)),
        )
        ruled_generator_pcurve = LineCurve(
            (0.0, 0.0),
            (0.0, 1.0),
            origin_exact=(Fraction(0), Fraction(0)),
            direction_exact=(Fraction(0), Fraction(1)),
        )
        ruled_plane_root = _ruled_chart_endpoint_root(
            ruled_plane_ring_pcurve, ruled_generator_pcurve, 0.0, 0.0
        )
        ruled_sphere_root = _ruled_chart_endpoint_root(
            ruled_sphere_ring_pcurve, ruled_generator_pcurve, 0.0, 1.0
        )
        ruled_region = SurfaceRegion(
            meridian.patch, np.asarray(((0.0, 0.0), (2 * pi, 1.0)))
        )
        ruled_sphere_lift = BRepCurveSurfaceLift(
            meridian.patch.second,
            meridian.patch,
            ruled_sphere_ring_pcurve,
            0.0,
            2 * pi,
            source_region=ruled_region,
        )
        plane_source_region = SurfaceRegion(
            seam.plane_source,
            np.asarray(builder.faces[seam.plane_face].bounds),
        )
        plane_source_lift = BRepCurveSurfaceLift(
            plane_curve,
            seam.plane_source,
            plane_pcurve,
            0.0,
            2 * pi,
            source_region=plane_source_region,
        )
        plane_lift = BRepCurveSurfaceLift(
            plane_curve,
            meridian.patch,
            ruled_plane_ring_pcurve,
            0.0,
            2 * pi,
            source_region=ruled_region,
        )
        plane_primary = builder.vertex_roots[vertex]
        if plane_primary is None:
            raise BRepBlendUnsupported(
                "closed seam chamfer: plane rail has no exact source vertex"
            )
        builder.vertex_roots[vertex] = BRepVertexRoot(
            plane_primary.primary,
            aliases=(BRepRootSupport(meridian.patch, ruled_plane_root),),
            source_edge_lifts=(plane_source_lift, plane_lift),
        )
    if isinstance(seam.plane_source, RevolutionSurface):
        if seam.plane_profile_affine is None or plane_profile_parameter_exact is None:
            raise BRepBlendUnsupported(
                "closed seam blend: Revolution cap has no exact radial trim map"
            )
        plane_counts: dict[int, int] = {}
        for loop in geometry.face_loops[seam.plane_face]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                plane_counts[edge] = plane_counts.get(edge, 0) + 1
        radial_edges = [edge for edge, count in plane_counts.items() if count == 2]
        if radial_edges:
            if len(radial_edges) != 1 or len(geometry.face_loops[seam.plane_face]) != 1:
                raise BRepBlendUnsupported(
                    "closed seam blend: Revolution cap radial trim is ambiguous"
                )
            radial_edge = radial_edges[0]
            radial_curve_index = builder.edge_curves[radial_edge]
            radial_curve = (
                None if radial_curve_index < 0 else builder.curves[radial_curve_index]
            )
            radial_start, radial_end = builder.edge_vertices[radial_edge]
            if not isinstance(radial_curve, LineCurve) or vertex not in (
                radial_start,
                radial_end,
            ):
                raise BRepBlendUnsupported(
                    "closed seam blend: Revolution cap radial edge is not an exact line"
                )
            center_vertex = radial_end if radial_start == vertex else radial_start
            center_exact = plane_curve.center_exact
            source_center_exact = geometry.vertex_exact_points[center_vertex]
            if source_center_exact is not None and source_center_exact != center_exact:
                raise BRepBlendUnsupported(
                    "closed seam blend: Revolution cap axis point changed source identity"
                )
            radial_origin_exact = (
                plane_point_exact if radial_start == vertex else center_exact
            )
            radial_direction_exact = tuple(
                (center_exact[index] - plane_point_exact[index])
                if radial_start == vertex
                else (plane_point_exact[index] - center_exact[index])
                for index in range(3)
            )
            radial_curve = LineCurve(
                np.asarray(tuple(float(value) for value in radial_origin_exact)),
                np.asarray(tuple(float(value) for value in radial_direction_exact)),
                origin_exact=radial_origin_exact,
                direction_exact=radial_direction_exact,
            )
            builder.curves[radial_curve_index] = radial_curve
            radial_roots = list(builder.edge_endpoint_roots[radial_edge])
            for endpoint, endpoint_vertex in enumerate((radial_start, radial_end)):
                if endpoint_vertex == vertex:
                    radial_parameter = Fraction.from_float(
                        float(builder.edge_ranges[radial_edge][endpoint])
                    )
                    radial_roots[endpoint] = ConicParameterEndpoint(
                        plane_vertex_conic,
                        radial_curve,
                        radial_parameter,
                        plane_point_exact,
                    )
            builder.edge_endpoint_roots[radial_edge] = (radial_roots[0], radial_roots[1])
            center_profile_parameter = seam.plane_profile_affine[1]
            old_outer_parameter = (
                seam.plane_profile_affine[0] * seam.seam_radius_exact
                + center_profile_parameter
            )
            for coedge in geometry.face_loops[seam.plane_face][0]:
                if geometry.coedge_edges[coedge] != radial_edge:
                    continue
                original_pcurve = builder.pcurves[coedge]
                # The forward seam use lives on the closing native-period sheet; the
                # radial remap acts on its source line and keeps the period gauge.
                original_line = (
                    original_pcurve.source_curve
                    if isinstance(original_pcurve, PeriodicPCurve)
                    else original_pcurve
                )
                if not isinstance(original_line, LineCurve):
                    raise BRepBlendUnsupported(
                        "closed seam blend: Revolution cap seam p-curve is not linear"
                    )
                pcurve_origin_exact = original_line.origin_exact
                pcurve_direction_exact = original_line.direction_exact
                pcurve_end_exact = tuple(
                    first + delta
                    for first, delta in zip(
                        pcurve_origin_exact, pcurve_direction_exact, strict=True
                    )
                )
                if (
                    pcurve_origin_exact[1] == center_profile_parameter
                    and pcurve_end_exact[1] == old_outer_parameter
                ):
                    new_pcurve_origin = (
                        pcurve_origin_exact[0],
                        center_profile_parameter,
                    )
                    new_pcurve_direction = (
                        pcurve_direction_exact[0],
                        plane_profile_parameter_exact - center_profile_parameter,
                    )
                elif (
                    pcurve_origin_exact[1] == old_outer_parameter
                    and pcurve_end_exact[1] == center_profile_parameter
                ):
                    new_pcurve_origin = (
                        pcurve_origin_exact[0],
                        plane_profile_parameter_exact,
                    )
                    new_pcurve_direction = (
                        pcurve_direction_exact[0],
                        center_profile_parameter - plane_profile_parameter_exact,
                    )
                else:
                    raise BRepBlendUnsupported(
                        "closed seam blend: Revolution cap seam p-curve has an unknown radial map"
                    )
                new_line = LineCurve(
                    np.asarray(tuple(float(value) for value in new_pcurve_origin)),
                    np.asarray(tuple(float(value) for value in new_pcurve_direction)),
                    origin_exact=new_pcurve_origin,
                    direction_exact=new_pcurve_direction,
                )
                new_pcurve = (
                    PeriodicPCurve(
                        new_line, original_pcurve.patch, original_pcurve.period_shifts
                    )
                    if isinstance(original_pcurve, PeriodicPCurve)
                    else new_line
                )
                builder.pcurves[coedge] = new_pcurve
                coedge_roots = list(builder.coedge_endpoint_roots[coedge])
                for endpoint in range(2):
                    parameter_exact = Fraction.from_float(
                        float(builder.edge_ranges[radial_edge][endpoint])
                    )
                    chart_point = tuple(
                        origin + parameter_exact * direction
                        for origin, direction in zip(
                            new_pcurve_origin,
                            new_pcurve_direction,
                            strict=True,
                        )
                    )
                    if chart_point == (
                        Fraction(0),
                        plane_profile_parameter_exact,
                    ):
                        coedge_roots[endpoint] = ConicParameterEndpoint(
                            plane_vertex_conic,
                            new_pcurve,
                            parameter_exact,
                            plane_point_exact,
                        )
                builder.coedge_endpoint_roots[coedge] = (
                    coedge_roots[0],
                    coedge_roots[1],
                )
            plane_face = builder.faces[seam.plane_face]
            plane_bounds = plane_face.bounds.copy()
            center_profile_low, center_profile_high = _exact_scalar_bounds(
                center_profile_parameter
            )
            plane_profile_low, plane_profile_high = _exact_scalar_bounds(
                plane_profile_parameter_exact
            )
            plane_bounds[:, 1] = (
                min(center_profile_low, plane_profile_low),
                max(center_profile_high, plane_profile_high),
            )
            plane_face.bounds = plane_bounds
    if seam.curved_profile_affine is None:
        curved_profile_parameter_exact = None
    elif meridian.curved_z_exact is None or isinstance(seam.curved, SpherePatch):
        curved_profile_parameter_exact = None
    else:
        curved_profile_parameter_exact = (
            seam.curved_profile_affine[0] * meridian.curved_z_exact
            + seam.curved_profile_affine[1]
        )
    curved_profile_parameter = (
        float(curved_profile_parameter_exact)
        if curved_profile_parameter_exact is not None
        else float(
            np.asarray(
                _mapped_profile_parameter(
                    seam.curved_profile_affine, meridian.curved_parameter
                )
            )
        )
    )
    contact_profile_root: ConicParameterRoot | None = None
    contact_chart_root: TrimIntersectionRoot | None = None
    if meridian.curved_radius_exact is not None and meridian.curved_z_exact is not None:
        curved_circle = _circle(
            seam,
            meridian.curved_radius,
            meridian.curved_radius_exact,
            axial_exact=meridian.curved_z_exact,
        )
        curved_curve: BRepCurve = curved_circle
        contact_point_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
            curved_circle.center_exact[0]
            + curved_circle.radius_exact * curved_circle.first_axis_exact[0],
            curved_circle.center_exact[1]
            + curved_circle.radius_exact * curved_circle.first_axis_exact[1],
            curved_circle.center_exact[2]
            + curved_circle.radius_exact * curved_circle.first_axis_exact[2],
        )
        point = np.asarray(
            tuple(float(value) for value in contact_point_exact), dtype=np.float64
        )
        if seam.curved_profile_circle is not None:
            profile_root = ConicParameterRoot.from_exact_point(
                seam.curved_profile_circle,
                contact_point_exact,
                curved_profile_parameter,
            )
            if profile_root is None:
                raise BRepBlendUnsupported(
                    "closed seam blend: exact sphere contact root is unavailable"
                )
            contact_profile_root = profile_root
        else:
            contact_profile_root = ConicParameterRoot(
                curved_circle, Fraction(1), Fraction(0)
            )
        contact_vertex_root = BRepVertexRoot(BRepConicVertex(contact_profile_root))
        cv = builder.vertex(
            point,
            exact_point=(
                (contact_point_exact[0], contact_point_exact[1], contact_point_exact[2])
                if (
                    isinstance(contact_point_exact[0], Fraction)
                    and isinstance(contact_point_exact[1], Fraction)
                    and isinstance(contact_point_exact[2], Fraction)
                )
                else None
            ),
            root=contact_vertex_root,
        )
        profile_parameter_root = (
            curved_profile_parameter_exact
            if curved_profile_parameter_exact is not None
            else Fraction.from_float(curved_profile_parameter)
        )
        if seam.curved_profile_circle is not None:
            curved_pcurve = AffinePCurve(
                ConicRootLinePCurve(
                    contact_profile_root,
                    Fraction(0),
                    Fraction(1),
                    parameter_range=(0.0, 2 * pi),
                ),
                ((0, 1), (1, 0)),
                (0, 0),
            )
        else:
            curved_pcurve = LineCurve(
                (0.0, curved_profile_parameter),
                (1.0, 0.0),
                origin_exact=(Fraction(0), profile_parameter_root),
                direction_exact=(Fraction(1), Fraction(0)),
            )
    elif (
        isinstance(seam.curved_source, RevolutionSurface)
        and isinstance(meridian.patch, RuledSurface)
        and ruled_sphere_root is not None
        and ruled_sphere_lift is not None
    ):
        curved_curve = meridian.patch.second
        if not isinstance(curved_curve, SurfaceMappedCurve):
            raise BRepBlendUnsupported(
                "closed seam chamfer: ruled sphere rail lost its rooted source curve"
            )
        sphere_ring_pcurve = curved_curve.pcurve
        contact_chart_root = _revolution_profile_chart_root(
            sphere_ring_pcurve, meridian.contact_bounds
        )
        original = geometry.pcurves[seam.curved_coedge]
        shifts = (
            original.period_shifts if isinstance(original, PeriodicPCurve) else (0, 0)
        )
        curved_pcurve = PeriodicPCurve(sphere_ring_pcurve, seam.curved_source, shifts)
        source_sphere_region = SurfaceRegion(
            seam.curved_source,
            np.asarray(builder.faces[seam.curved_face].bounds),
        )
        source_sphere_lift = BRepCurveSurfaceLift(
            curved_curve,
            seam.curved_source,
            sphere_ring_pcurve,
            0.0,
            2 * pi,
            source_region=source_sphere_region,
        )
        point = np.asarray(
            curved_curve.evaluate(jnp.asarray(0.0, dtype=jnp.float64)),
            dtype=np.float64,
        )
        cv = builder.vertex(
            point,
            root=BRepVertexRoot(
                BRepRootSupport(seam.curved_source, contact_chart_root),
                aliases=(BRepRootSupport(meridian.patch, ruled_sphere_root),),
                source_edge_lifts=(source_sphere_lift, ruled_sphere_lift),
            ),
        )
    else:
        point = np.asarray(
            seam.curved_source.evaluate(
                jnp.asarray((0.0, curved_profile_parameter), dtype=jnp.float64)
            ),
            dtype=np.float64,
        )
        cv = builder.vertex(point)
        curved_curve = SurfaceIsoparametricCurve(
            seam.curved_source, 1, curved_profile_parameter
        )
        curved_pcurve = _isoparametric_pcurve(curved_curve)
    curved_roots = (
        NativePeriodEndpoint(curved_curve, seam.curved_source, 0, rational=0, turns=0),
        NativePeriodEndpoint(curved_curve, seam.curved_source, 0, rational=0, turns=1),
    )
    if contact_chart_root is None:
        curved_coedge_roots = (
            NativePeriodEndpoint(
                curved_pcurve, seam.curved_source, 0, rational=0, turns=0
            ),
            NativePeriodEndpoint(
                curved_pcurve, seam.curved_source, 0, rational=0, turns=1
            ),
        )
    else:
        root_parameter = float(np.asarray(contact_chart_root.parameters)[0])
        curved_region = SurfaceRegion(
            seam.curved_source,
            np.asarray(builder.faces[seam.curved_face].bounds),
        )
        curved_coedge_roots = (
            TrimRootEndpoint(
                contact_chart_root,
                "first",
                affine_parameter_offset=-root_parameter,
                source_pcurve=curved_pcurve,
                source_surface=curved_region,
            ),
            TrimRootEndpoint(
                contact_chart_root,
                "first",
                affine_parameter_offset=2 * pi - root_parameter,
                source_pcurve=curved_pcurve,
                source_surface=curved_region,
            ),
        )
    ce = builder.edge(
        curved_curve,
        0.0,
        2 * pi,
        cv,
        cv,
        endpoint_roots=curved_roots,
    )
    builder.coedge_edges[seam.curved_coedge] = ce
    builder.pcurves[seam.curved_coedge] = curved_pcurve
    builder.coedge_endpoint_roots[seam.curved_coedge] = curved_coedge_roots
    face = builder.faces[seam.curved_face]
    box = face.bounds.copy()
    endpoint = seam.curved_endpoint
    if seam.curved_profile_circle is not None and contact_profile_root is not None:
        contact_chart_bounds = (
            contact_profile_root.parameter_lower,
            contact_profile_root.parameter_upper,
        )
    elif contact_chart_root is not None:
        # The isolating box is not the latitude enclosure of its unique root.
        contact_chart_bounds = meridian.contact_bounds
    elif curved_profile_parameter_exact is not None:
        contact_chart_bounds = _exact_scalar_bounds(curved_profile_parameter_exact)
    else:
        contact_chart_bounds = (
            curved_profile_parameter,
            curved_profile_parameter,
        )
    box[endpoint, 1] = contact_chart_bounds[endpoint]
    face.bounds = box
    start, end = builder.edge_vertices[seam.seam_edge]
    builder.edge_vertices[seam.seam_edge] = (cv, end) if endpoint == 0 else (start, cv)
    if isinstance(seam.curved_source, RevolutionSurface):
        profile_curve = seam.curved_source.curve
        original_edge_roots = geometry.edge_endpoint_roots[seam.seam_edge]
        if isinstance(profile_curve, LineCurve):
            if curved_profile_parameter_exact is None:
                raise BRepBlendUnsupported(
                    "closed seam blend: exact line-profile contact parameter is unavailable"
                )
            profile_parameters: list[ExactScalar] = []
            for index, root in enumerate(original_edge_roots):
                if index == endpoint:
                    profile_parameters.append(curved_profile_parameter_exact)
                elif (
                    isinstance(root, ConicParameterEndpoint)
                    and root.parameter_exact is not None
                ):
                    profile_parameters.append(root.parameter_exact)
                else:
                    profile_parameters.append(
                        Fraction.from_float(
                            float(geometry.edge_ranges[seam.seam_edge][index])
                        )
                    )
            start_profile, end_profile = profile_parameters
            start_point = tuple(
                profile_curve.origin_exact[axis]
                + start_profile * profile_curve.direction_exact[axis]
                for axis in range(3)
            )
            line_direction = tuple(
                (end_profile - start_profile) * profile_curve.direction_exact[axis]
                for axis in range(3)
            )
            seam_curve: BRepCurve = LineCurve(
                np.asarray(tuple(float(value) for value in start_point)),
                np.asarray(tuple(float(value) for value in line_direction)),
                origin_exact=start_point,
                direction_exact=line_direction,
            )
            builder.curves[builder.edge_curves[seam.seam_edge]] = seam_curve
            builder.edge_ranges[seam.seam_edge] = (0.0, 1.0)
            edge_roots = []
            for index, root in enumerate(original_edge_roots):
                if index == endpoint:
                    edge_roots.append(
                        None
                        if contact_profile_root is None
                        else ConicParameterEndpoint(
                            contact_profile_root,
                            seam_curve,
                            Fraction(index),
                            contact_point_exact,
                        )
                    )
                elif isinstance(root, ConicParameterEndpoint):
                    edge_roots.append(
                        ConicParameterEndpoint(
                            root.root,
                            seam_curve,
                            Fraction(index),
                            root.point_exact,
                        )
                    )
                else:
                    edge_roots.append(root)
            builder.edge_endpoint_roots[seam.seam_edge] = (edge_roots[0], edge_roots[1])
            for coedge in geometry.face_loops[seam.curved_face][0]:
                if builder.coedge_edges[coedge] != seam.seam_edge:
                    continue
                original = geometry.pcurves[coedge]
                shifts = (
                    original.period_shifts
                    if isinstance(original, PeriodicPCurve)
                    else (0, 0)
                )
                source_pcurve = LineCurve(
                    (0.0, float(start_profile)),
                    (0.0, float(end_profile - start_profile)),
                    origin_exact=(Fraction(0), start_profile),
                    direction_exact=(Fraction(0), end_profile - start_profile),
                )
                pcurve = PeriodicPCurve(source_pcurve, seam.curved_source, shifts)
                builder.pcurves[coedge] = pcurve
                coedge_roots = []
                for index, root in enumerate(geometry.coedge_endpoint_roots[coedge]):
                    if index == endpoint:
                        coedge_roots.append(
                            None
                            if contact_profile_root is None
                            else ConicParameterEndpoint(
                                contact_profile_root,
                                pcurve,
                                Fraction(index),
                                contact_point_exact,
                            )
                        )
                    elif isinstance(root, ConicParameterEndpoint):
                        coedge_roots.append(
                            ConicParameterEndpoint(
                                root.root,
                                pcurve,
                                Fraction(index),
                                root.point_exact,
                            )
                        )
                    else:
                        coedge_roots.append(root)
                builder.coedge_endpoint_roots[coedge] = (coedge_roots[0], coedge_roots[1])
        elif isinstance(profile_curve, CircleCurve) and contact_profile_root is not None:
            seam_curve = profile_curve
            builder.curves[builder.edge_curves[seam.seam_edge]] = seam_curve
            builder.edge_ranges[seam.seam_edge] = (
                float(box[0, 1]),
                float(box[1, 1]),
            )
            edge_roots = []
            for index, root in enumerate(original_edge_roots):
                if index == endpoint:
                    edge_roots.append(
                        ConicParameterEndpoint(
                            contact_profile_root,
                            seam_curve,
                            None,
                            contact_point_exact,
                        )
                    )
                else:
                    edge_roots.append(root)
            builder.edge_endpoint_roots[seam.seam_edge] = (edge_roots[0], edge_roots[1])
            for coedge in geometry.face_loops[seam.curved_face][0]:
                if builder.coedge_edges[coedge] != seam.seam_edge:
                    continue
                original = geometry.pcurves[coedge]
                shifts = (
                    original.period_shifts
                    if isinstance(original, PeriodicPCurve)
                    else (0, 0)
                )
                source_pcurve = LineCurve(
                    (0.0, 0.0),
                    (0.0, 1.0),
                    origin_exact=(Fraction(0), Fraction(0)),
                    direction_exact=(Fraction(0), Fraction(1)),
                )
                pcurve = PeriodicPCurve(source_pcurve, seam.curved_source, shifts)
                builder.pcurves[coedge] = pcurve
                coedge_roots = []
                for index, root in enumerate(geometry.coedge_endpoint_roots[coedge]):
                    if index == endpoint:
                        coedge_roots.append(
                            ConicParameterEndpoint(
                                contact_profile_root,
                                pcurve,
                                None,
                                contact_point_exact,
                            )
                        )
                    elif isinstance(root, ConicParameterEndpoint):
                        coedge_roots.append(
                            ConicParameterEndpoint(
                                root.root,
                                pcurve,
                                root.parameter_exact,
                                root.point_exact,
                            )
                        )
                    else:
                        coedge_roots.append(root)
                builder.coedge_endpoint_roots[coedge] = (coedge_roots[0], coedge_roots[1])
        elif isinstance(profile_curve, CircleCurve) and contact_chart_root is not None:
            seam_curve = profile_curve
            builder.curves[builder.edge_curves[seam.seam_edge]] = seam_curve
            builder.edge_ranges[seam.seam_edge] = (
                float(box[0, 1]),
                float(box[1, 1]),
            )
            edge_roots = []
            for index, root in enumerate(original_edge_roots):
                if index == endpoint:
                    edge_roots.append(TrimRootEndpoint(contact_chart_root, "second"))
                elif isinstance(root, ConicParameterEndpoint):
                    edge_roots.append(
                        ConicParameterEndpoint(
                            root.root,
                            seam_curve,
                            root.parameter_exact,
                            root.point_exact,
                        )
                    )
                else:
                    edge_roots.append(root)
            builder.edge_endpoint_roots[seam.seam_edge] = (
                edge_roots[0],
                edge_roots[1],
            )
            for coedge in geometry.face_loops[seam.curved_face][0]:
                if builder.coedge_edges[coedge] != seam.seam_edge:
                    continue
                original = geometry.pcurves[coedge]
                shifts = (
                    original.period_shifts
                    if isinstance(original, PeriodicPCurve)
                    else (0, 0)
                )
                source_pcurve = LineCurve(
                    (0.0, 0.0),
                    (0.0, 1.0),
                    origin_exact=(Fraction(0), Fraction(0)),
                    direction_exact=(Fraction(0), Fraction(1)),
                )
                pcurve = PeriodicPCurve(source_pcurve, seam.curved_source, shifts)
                builder.pcurves[coedge] = pcurve
                face_region = SurfaceRegion(
                    seam.curved_source,
                    np.asarray(builder.faces[seam.curved_face].bounds),
                )
                coedge_roots = []
                for index, root in enumerate(geometry.coedge_endpoint_roots[coedge]):
                    if index == endpoint:
                        coedge_roots.append(
                            TrimRootEndpoint(
                                contact_chart_root,
                                "second",
                                source_pcurve=pcurve,
                                source_surface=face_region,
                            )
                        )
                    elif isinstance(root, ConicParameterEndpoint):
                        coedge_roots.append(
                            ConicParameterEndpoint(
                                root.root,
                                pcurve,
                                root.parameter_exact,
                                root.point_exact,
                            )
                        )
                    else:
                        coedge_roots.append(root)
                builder.coedge_endpoint_roots[coedge] = (
                    coedge_roots[0],
                    coedge_roots[1],
                )
        elif isinstance(profile_curve, CircleCurve):
            raise BRepBlendUnsupported(
                "closed seam blend: exact Revolution circle-profile contact root is unavailable"
            )
        else:
            raise BRepBlendUnsupported(
                "closed seam blend: exact Revolution meridian edge type is unsupported"
            )
    else:
        builder.edge_ranges[seam.seam_edge] = (float(box[0, 1]), float(box[1, 1]))
        builder.edge_endpoint_roots[seam.seam_edge] = (None, None)
        builder.curves[builder.edge_curves[seam.seam_edge]] = SurfaceIsoparametricCurve(
            seam.curved_source, 0, 0.0
        )
        for coedge in geometry.face_loops[seam.curved_face][0]:
            if builder.coedge_edges[coedge] == seam.seam_edge:
                original = geometry.pcurves[coedge]
                shifts = (
                    original.period_shifts
                    if isinstance(original, PeriodicPCurve)
                    else (0, 0)
                )
                builder.pcurves[coedge] = PeriodicPCurve(
                    LineCurve((0.0, 0.0), (0.0, 1.0)), seam.curved_source, shifts
                )
                builder.coedge_endpoint_roots[coedge] = (None, None)
    return seam.edge, ce


def _ruled_alias_root(
    vertex_root: BRepVertexRoot | None, patch: RuledSurface, /
) -> TrimIntersectionRoot:
    if vertex_root is not None:
        for support in vertex_root.aliases:
            if support.patch is patch and isinstance(support.root, TrimIntersectionRoot):
                return support.root
    raise BRepBlendUnsupported(
        "closed seam chamfer: ruled endpoint lacks its exact source-root alias"
    )


def _stage_blend(
    builder: _Builder,
    seam: _Seam,
    meridian: _Meridian,
    plane_edge: int,
    curved_edge: int,
    /,
) -> int:
    patch = meridian.patch
    face_lower, face_upper = meridian.lower, meridian.upper
    lower_edge, upper_edge = (
        (plane_edge, curved_edge)
        if meridian.plane_parameter < meridian.blend_parameter
        else (curved_edge, plane_edge)
    )
    start = builder.edge_vertices[lower_edge][0]
    end = builder.edge_vertices[upper_edge][0]
    if isinstance(patch, TorusPatch):
        if (
            meridian.plane_radius_exact is None
            or meridian.curved_radius_exact is None
            or meridian.curved_z_exact is None
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: exact torus boundary circle is unavailable"
            )
        center_exact = tuple(
            patch.center_exact[index]
            + patch.major_radius_exact * patch.first_axis_exact[index]
            for index in range(3)
        )
        cross_circle = CircleCurve(
            np.asarray(tuple(float(value) for value in center_exact), dtype=np.float64),
            patch.first_axis,
            patch.axis,
            patch.minor_radius,
            center_exact=center_exact,
            first_axis_exact=patch.first_axis_exact,
            second_axis_exact=patch.axis_exact,
            radius_exact=patch.minor_radius_exact,
        )
        contact_point_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
            seam.axis_origin_exact[0]
            + meridian.curved_z_exact * seam.axis_exact[0]
            + meridian.curved_radius_exact * seam.first_axis_exact[0],
            seam.axis_origin_exact[1]
            + meridian.curved_z_exact * seam.axis_exact[1]
            + meridian.curved_radius_exact * seam.first_axis_exact[1],
            seam.axis_origin_exact[2]
            + meridian.curved_z_exact * seam.axis_exact[2]
            + meridian.curved_radius_exact * seam.first_axis_exact[2],
        )
        curved_vertex = builder.edge_vertices[curved_edge][0]
        plane_vertex = builder.edge_vertices[plane_edge][0]
        plane_vertex_root = builder.vertex_roots[plane_vertex]
        if plane_vertex_root is None or not isinstance(
            plane_vertex_root.primary, BRepConicVertex
        ):
            raise BRepBlendUnsupported(
                "closed seam blend: plane contact lacks its exact conic vertex"
            )
        plane_point_exact = plane_vertex_root.primary.point_exact
        plane_cross_root = ConicParameterRoot.from_exact_point(
            cross_circle, plane_point_exact, meridian.plane_parameter
        )
        if plane_cross_root is None:
            raise BRepBlendUnsupported(
                "closed seam blend: plane contact is not on the exact torus section"
            )
        root_data: dict[
            int,
            tuple[
                ConicParameterRoot,
                tuple[ExactScalar, ExactScalar, ExactScalar],
            ],
        ] = {plane_vertex: (plane_cross_root, plane_point_exact)}
        contact_cross_root = ConicParameterRoot.from_exact_point(
            cross_circle, contact_point_exact, meridian.blend_parameter
        )
        if contact_cross_root is None:
            raise BRepBlendUnsupported(
                "closed seam blend: curved contact is not on the exact torus section"
            )
        if builder.vertex_roots[curved_vertex] is None:
            builder.vertex_roots[curved_vertex] = BRepVertexRoot(
                BRepConicVertex(contact_cross_root)
            )
        root_data[curved_vertex] = (contact_cross_root, contact_point_exact)
        start_root, start_point = root_data[start]
        end_root, end_point = root_data[end]
        edge_roots = (
            ConicParameterEndpoint(start_root, cross_circle, None, start_point),
            ConicParameterEndpoint(end_root, cross_circle, None, end_point),
        )
        edge = builder.edge(
            cross_circle,
            meridian.lower,
            meridian.upper,
            start,
            end,
            endpoint_roots=edge_roots,
        )
        ring_pcurves: list[AbstractCurve] = []
        for ring_root in (start_root, end_root):
            ring_pcurves.append(
                AffinePCurve(
                    ConicRootLinePCurve(
                        ring_root,
                        Fraction(0),
                        Fraction(1),
                        parameter_range=(0.0, 2 * pi),
                    ),
                    ((0, 1), (1, 0)),
                    (0, 0),
                )
            )
            face_lower = min(face_lower, ring_root.parameter_lower)
            face_upper = max(face_upper, ring_root.parameter_upper)
        low_pcurve, high_pcurve = ring_pcurves
        low_roots = (
            NativePeriodEndpoint(low_pcurve, patch, 0, rational=0, turns=0),
            NativePeriodEndpoint(low_pcurve, patch, 0, rational=0, turns=1),
        )
        high_roots = (
            NativePeriodEndpoint(high_pcurve, patch, 0, rational=0, turns=0),
            NativePeriodEndpoint(high_pcurve, patch, 0, rational=0, turns=1),
        )
        low = builder.coedge(lower_edge, 1, low_pcurve, endpoint_roots=low_roots)
        right_pcurve = LineCurve((0.0, 0.0), (0.0, 1.0))
        right_carrier = PeriodicPCurve(right_pcurve, patch, (1, 0))
        right_roots = (
            ConicParameterEndpoint(start_root, right_carrier, None, start_point),
            ConicParameterEndpoint(end_root, right_carrier, None, end_point),
        )
        right = builder.coedge(edge, 1, right_carrier, endpoint_roots=right_roots)
        high = builder.coedge(upper_edge, -1, high_pcurve, endpoint_roots=high_roots)
        left_carrier = PeriodicPCurve(right_pcurve, patch, (0, 0))
        left_roots = (
            ConicParameterEndpoint(start_root, left_carrier, None, start_point),
            ConicParameterEndpoint(end_root, left_carrier, None, end_point),
        )
        left = builder.coedge(edge, -1, left_carrier, endpoint_roots=left_roots)
    elif isinstance(patch, RuledSurface):
        lower_face, upper_face = (
            (seam.plane_face, seam.curved_face)
            if lower_edge == plane_edge
            else (seam.curved_face, seam.plane_face)
        )

        def opposite_boundary_sense(edge: int, face: int, /) -> int:
            coedges = tuple(
                coedge
                for loop in builder.faces[face].loops
                for coedge in loop
                if builder.coedge_edges[coedge] == edge
            )
            if len(coedges) != 1:
                raise BRepBlendUnsupported(
                    "closed seam chamfer: ruled rail needs a unique oriented source boundary"
                )
            coedge = coedges[0]
            return (
                -builder.coedge_senses[coedge]
                * builder.faces[face].orientation
                * seam.radial_sense
            )

        lower_sense = opposite_boundary_sense(lower_edge, lower_face)
        upper_sense = opposite_boundary_sense(upper_edge, upper_face)
        plane_root = _ruled_alias_root(builder.vertex_roots[start], patch)
        sphere_root = _ruled_alias_root(builder.vertex_roots[end], patch)
        plane_ring = LineCurve(
            (0.0, 0.0),
            (1.0, 0.0),
            origin_exact=(Fraction(0), Fraction(0)),
            direction_exact=(Fraction(1), Fraction(0)),
        )
        sphere_ring = LineCurve(
            (0.0, 1.0),
            (1.0, 0.0),
            origin_exact=(Fraction(0), Fraction(1)),
            direction_exact=(Fraction(1), Fraction(0)),
        )
        generator = LineCurve(
            (0.0, 0.0),
            (0.0, 1.0),
            origin_exact=(Fraction(0), Fraction(0)),
            direction_exact=(Fraction(0), Fraction(1)),
        )
        ruled_region = SurfaceRegion(patch, np.asarray(((0.0, 0.0), (2 * pi, 1.0))))

        def root_endpoint(
            root: TrimIntersectionRoot,
            operand: Literal["first", "second"],
            target_parameter: float,
            pcurve: BRepCurve | None = None,
            /,
        ) -> TrimRootEndpoint:
            column = 0 if operand == "first" else 1
            root_parameter = float(np.asarray(root.parameters)[column])
            return TrimRootEndpoint(
                root,
                operand,
                affine_parameter_offset=target_parameter - root_parameter,
                source_pcurve=pcurve,
                source_surface=None if pcurve is None else ruled_region,
            )

        edge = builder.edge(
            SurfaceMappedCurve(patch, generator),
            0.0,
            1.0,
            start,
            end,
            endpoint_roots=(
                root_endpoint(plane_root, "second", 0.0),
                root_endpoint(sphere_root, "second", 1.0),
            ),
        )
        plane_ring_roots = (
            root_endpoint(plane_root, "first", 0.0, plane_ring),
            root_endpoint(plane_root, "first", 2 * pi, plane_ring),
        )
        sphere_ring_roots = (
            root_endpoint(sphere_root, "first", 0.0, sphere_ring),
            root_endpoint(sphere_root, "first", 2 * pi, sphere_ring),
        )
        low = builder.coedge(
            lower_edge, lower_sense, plane_ring, endpoint_roots=plane_ring_roots
        )
        right_pcurve = PeriodicPCurve(generator, patch, (1, 0))
        right_roots = (
            root_endpoint(plane_root, "second", 0.0, right_pcurve),
            root_endpoint(sphere_root, "second", 1.0, right_pcurve),
        )
        right = builder.coedge(edge, 1, right_pcurve, endpoint_roots=right_roots)
        high = builder.coedge(
            upper_edge, upper_sense, sphere_ring, endpoint_roots=sphere_ring_roots
        )
        left_pcurve = PeriodicPCurve(generator, patch, (0, 0))
        left_roots = (
            root_endpoint(plane_root, "second", 0.0, left_pcurve),
            root_endpoint(sphere_root, "second", 1.0, left_pcurve),
        )
        left = builder.coedge(edge, -1, left_pcurve, endpoint_roots=left_roots)
        face_bounds = np.asarray(((0.0, 0.0), (2 * pi, 1.0)))
        return builder.face(
            patch,
            face_bounds,
            seam.radial_sense,
            "chamfer",
            [[low, right, high, left]],
        )
    else:
        if not isinstance(patch, RevolutionSurface) or not isinstance(
            patch.curve, LineCurve
        ):
            raise BRepBlendUnsupported(
                "closed seam chamfer: exact normalized generator is unavailable"
            )
        curve = patch.curve
        start_binding = builder.vertex_roots[start]
        end_binding = builder.vertex_roots[end]
        if (
            start_binding is None
            or end_binding is None
            or not isinstance(start_binding.primary, BRepConicVertex)
            or not isinstance(end_binding.primary, BRepConicVertex)
        ):
            raise BRepBlendUnsupported(
                "closed seam chamfer: generator contacts lack exact conic vertices"
            )
        first_conic, last_conic = start_binding.primary, end_binding.primary
        edge_roots = (
            ConicParameterEndpoint(
                first_conic.root, curve, Fraction(0), first_conic.point_exact
            ),
            ConicParameterEndpoint(
                last_conic.root, curve, Fraction(1), last_conic.point_exact
            ),
        )
        edge = builder.edge(curve, 0.0, 1.0, start, end, endpoint_roots=edge_roots)
        pcurve = LineCurve((0.0, 0.0), (0.0, 1.0))
        low_pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))
        high_pcurve = LineCurve((0.0, 1.0), (1.0, 0.0))
        low = builder.coedge(
            lower_edge,
            1,
            low_pcurve,
            endpoint_roots=(
                NativePeriodEndpoint(low_pcurve, patch, 0, rational=0, turns=0),
                NativePeriodEndpoint(low_pcurve, patch, 0, rational=0, turns=1),
            ),
        )
        high = builder.coedge(
            upper_edge,
            -1,
            high_pcurve,
            endpoint_roots=(
                NativePeriodEndpoint(high_pcurve, patch, 0, rational=0, turns=0),
                NativePeriodEndpoint(high_pcurve, patch, 0, rational=0, turns=1),
            ),
        )
        right_pcurve = PeriodicPCurve(pcurve, patch, (1, 0))
        left_pcurve = PeriodicPCurve(pcurve, patch, (0, 0))
        right = builder.coedge(
            edge,
            1,
            right_pcurve,
            endpoint_roots=(
                ConicParameterEndpoint(
                    first_conic.root, right_pcurve, Fraction(0), first_conic.point_exact
                ),
                ConicParameterEndpoint(
                    last_conic.root, right_pcurve, Fraction(1), last_conic.point_exact
                ),
            ),
        )
        left = builder.coedge(
            edge,
            -1,
            left_pcurve,
            endpoint_roots=(
                ConicParameterEndpoint(
                    first_conic.root, left_pcurve, Fraction(0), first_conic.point_exact
                ),
                ConicParameterEndpoint(
                    last_conic.root, left_pcurve, Fraction(1), last_conic.point_exact
                ),
            ),
        )
    if isinstance(patch, TorusPatch):
        orientation = 1
    else:
        if meridian.curved_z_exact is None:
            raise BRepBlendUnsupported(
                "Exact chamfer orientation lacks its contact height."
            )
        orientation = seam.radial_sense * (
            1 if meridian.curved_z_exact > seam.z_exact else -1
        )
    return builder.face(
        patch,
        np.asarray(((0.0, face_lower), (2 * pi, face_upper))),
        orientation,
        "fillet" if isinstance(patch, TorusPatch) else "chamfer",
        [[low, right, high, left]],
    )


def closed_seam_blend(
    model: BRepModel,
    edge_index: int,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
) -> BRepModel:
    """Replace one full circular plane/quadric edge and publish native topology."""
    if (radius is None) == (setbacks is None):
        raise ValueError(
            "closed_seam_blend.specification must select one radius or setback pair; got an ambiguous selection"
        )
    seam = _seam(model, edge_index)
    if radius is not None:
        meridian = _fillet(seam, radius)
    elif setbacks is not None:
        meridian = _chamfer(seam, setbacks)
    else:
        raise ValueError(
            "closed_seam_blend.specification must select a blend; got no specification"
        )
    _require_clearance(model, seam, meridian)
    geometry = model.geometry
    builder = _copy_builder(model)
    plane_edge, curved_edge = _stage_boundaries(builder, geometry, seam, meridian)
    blend = _stage_blend(builder, seam, meridian, plane_edge, curved_edge)
    shells = tuple(
        tuple((*faces, blend)) if seam.curved_face in faces else faces
        for faces in geometry.shell_faces
    )
    orientations = tuple(
        tuple((*row, 1)) if seam.curved_face in faces else row
        for faces, row in zip(
            geometry.shell_faces, geometry.shell_orientations, strict=True
        )
    )
    geometry_, patches, bounds, signs, tags, trims = builder.geometry(
        solid=bool(geometry.solid_shells),
        shell_faces=shells,
        shell_orientations=orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )
    return assemble_brep_model(
        geometry_,
        patches,
        bounds,
        signs,
        tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=trims,
        curve_surface_tolerance=model.report.curve_surface_tolerance,
        curve_surface_scale=model.report.curve_surface_scale,
    )


def closed_seam_tangency_certificate(
    model: BRepModel,
    edge_index: int,
    radius: float,
    /,
) -> ClosedSeamTangencyCertificate:
    """Enclose analytic normals at both whole-circle fillet boundaries."""
    seam = _seam(model, edge_index)
    slope_bounds = _cone_slope_bounds(seam)

    meridian = _fillet(seam, radius)

    def differences(point: Array) -> Array:
        torus_plane = jnp.stack((jnp.cos(point[0]), jnp.sin(point[0])))
        torus_curved = jnp.stack((jnp.cos(point[1]), jnp.sin(point[1])))
        match seam.curved:
            case CylinderPatch():
                normal = jnp.asarray((seam.radial_sense, 0.0), dtype=jnp.float64)
            case ConePatch():
                slope = _cone_slope(seam) if slope_bounds is None else point[2]
                normal = (
                    seam.radial_sense
                    * jnp.stack((jnp.ones_like(slope), -slope))
                    / jnp.sqrt(1 + slope * slope)
                )
            case SpherePatch():
                latitude = meridian.curved_parameter
                normal = seam.radial_sense * jnp.stack(
                    (jnp.cos(latitude), jnp.sin(latitude))
                )
        return jnp.stack(
            (torus_plane - jnp.asarray((0.0, seam.axial_sense)), torus_curved - normal)
        )

    prepared = prepare_interval_function(
        differences, 2 if slope_bounds is None else 3, batch_capacity=1
    )
    angles = np.asarray(
        ((meridian.plane_parameter, meridian.blend_parameter),), dtype=np.float64
    )
    if slope_bounds is None:
        angles_low = angles_high = angles
    else:
        slope_low, slope_high = slope_bounds
        angles_low = np.concatenate((angles, np.full((1, 1), slope_low)), axis=1)
        angles_high = np.concatenate((angles, np.full((1, 1), slope_high)), axis=1)
    low, high = prepared.evaluate(angles_low, angles_high)
    return ClosedSeamTangencyCertificate(
        jnp.asarray(np.stack((low[0], high[0]), axis=-1)),
        radius,
        seam.plane_face,
        seam.curved_face,
        meridian.plane_parameter,
        meridian.curved_parameter,
    )


def _angular_isoline_direction(curve: AbstractCurve, /) -> Array:
    if isinstance(curve, LineCurve):
        return curve.direction
    if isinstance(curve, AffinePCurve) and isinstance(curve.curve, ConicRootLinePCurve):
        return curve.matrix_value @ jnp.stack(
            (jnp.zeros_like(curve.curve.height_direction), curve.curve.height_direction)
        )
    raise ValueError(
        f"closed seam radius map requires an affine angular isoline; got {type(curve).__name__}"
    )


@final
class _ClosedSeamRadiusMap(AbstractBRepParameterMap):
    seam: _Seam
    radius: Array
    result: BRepModel
    source_model_id: str = eqx.field(static=True)
    edge_index: int = eqx.field(static=True)
    angle_shift: float = eqx.field(static=True)
    endpoint: int = eqx.field(static=True)
    other_radius: float = eqx.field(static=True)
    radius_upper: float | None = eqx.field(static=True)

    def compile(
        self, context: _ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        if model.model_id != self.result.model_id:
            raise ValueError(
                "closed_seam_radius_map.model must match its certified construction; got a different model identity"
            )
        binding = context.bind(
            ParameterId(
                f"{self.source_model_id}:edge:{self.edge_index}:fillet", "radius"
            ),
            self.radius,
            role="length",
            physical_scale=float(self.radius),
            bounds=(0.0, self.radius_upper),
            trainable=True,
        )
        plane_endpoint: int | None = None
        if self.seam.plane_profile_affine is not None:
            plane_template = model.geometry.pcurves[self.seam.plane_coedge]
            if isinstance(plane_template, LineCurve):
                plane_start = plane_template.origin_exact[1]
            elif isinstance(plane_template, ExactLineCurve):
                plane_start = plane_template.first_exact[1]
            else:
                raise BRepBlendUnsupported(
                    "closed seam radius map requires an affine source cap trim"
                )
            plane_bounds = np.asarray(model.parameter_bounds)[self.seam.plane_face]
            lower, upper = _exact_scalar_bounds(plane_start)
            lower_endpoint = lower == plane_bounds[0, 1]
            upper_endpoint = upper == plane_bounds[1, 1]
            if lower_endpoint == upper_endpoint:
                raise BRepBlendUnsupported(
                    "closed seam radius map cap trim is not a unique profile endpoint"
                )
            plane_endpoint = 0 if lower_endpoint else 1
        return _BoundClosedSeamRadiusMap(
            self.seam,
            self.result,
            binding,
            self.angle_shift,
            self.endpoint,
            self.other_radius,
            plane_endpoint,
        )


@final
class _BoundClosedSeamRadiusMap(AbstractBoundBRepParameterMap):
    seam: _Seam
    model: BRepModel
    binding: ParameterBinding = eqx.field(static=True)
    angle_shift: float = eqx.field(static=True)
    endpoint: int = eqx.field(static=True)
    other_radius: float = eqx.field(static=True)
    plane_endpoint: int | None = eqx.field(static=True)

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        from ._differentiable import (
            _RealizedConicCurve,
            _RealizedLineCurve,
            _RealizedTorusPatch,
            _RealizedTrimEdge,
        )

        radius = self.binding.read(state)
        rc, zc, rt, zt, parameter = _fillet_values(self.seam, radius)
        seam, model = self.seam, self.model
        geometry = model.geometry
        source_parameter = _mapped_profile_parameter(
            seam.curved_profile_affine, parameter
        )
        if model.tessellation is not None:
            raise BRepBlendUnsupported(
                "closed seam radius map cannot retain a tessellation fidelity certificate under parameter motion"
            )
        old_patch = model.patches[-1]
        if not isinstance(old_patch, TorusPatch):
            raise ValueError(
                f"closed_seam_radius_map.patch must be a torus; got {type(old_patch).__name__}"
            )
        origin = (
            seam.curved.center
            if isinstance(seam.curved, SpherePatch)
            else seam.curved.origin
        )
        patch = _RealizedTorusPatch(
            origin + zc * seam.curved.axis,
            old_patch.first_axis,
            old_patch.second_axis,
            old_patch.axis,
            rc,
            radius,
            periods=old_patch.periods,
            period_symbols=old_patch.native_period_symbols(),
        )
        patches = (*model.patches[:-1], patch)
        angle = jnp.arctan2(zt - zc, rt - rc) + self.angle_shift
        blend_loop = geometry.face_loops[-1][0]
        plane_blend_coedge = next(
            coedge for coedge in blend_loop if geometry.coedge_edges[coedge] == seam.edge
        )
        fixed_curve = geometry.pcurves[plane_blend_coedge]
        if not isinstance(fixed_curve, AffinePCurve) or not isinstance(
            fixed_curve.curve, ConicRootLinePCurve
        ):
            raise RuntimeError(
                "closed_seam_radius_map.plane_phase must retain its exact rooted chart"
            )
        fixed = fixed_curve.curve.angle
        lower, upper = jnp.minimum(angle, fixed), jnp.maximum(angle, fixed)
        bounds = model.parameter_bounds.at[seam.curved_face, self.endpoint, 1].set(
            source_parameter
        )
        plane_source_parameter = _mapped_profile_parameter(seam.plane_profile_affine, rc)
        if seam.plane_profile_affine is not None:
            if not isinstance(seam.plane_source, RevolutionSurface):
                raise RuntimeError("The dynamic cap chart lost its revolution source.")
            if self.plane_endpoint is None:
                raise RuntimeError("The prepared cap chart lost its endpoint binding.")
            bounds = bounds.at[seam.plane_face, self.plane_endpoint, 1].set(
                plane_source_parameter
            )
        pcurves = list(geometry.pcurves)
        plane_curve = pcurves[seam.plane_coedge]
        if seam.plane_profile_affine is None:
            if not isinstance(plane_curve, CircleCurve):
                raise ValueError(
                    f"closed_seam_radius_map.plane_trim must be a circle; got {type(plane_curve).__name__}"
                )
            pcurves[seam.plane_coedge] = _RealizedConicCurve(
                "circle",
                plane_curve.center,
                plane_curve.first_axis,
                plane_curve.second_axis,
                rc,
                parameter_domain=plane_curve.parameter_domain,
                period=plane_curve.period,
                period_symbol=plane_curve.native_period_symbol(),
            )
        elif isinstance(plane_curve, LineCurve):
            pcurves[seam.plane_coedge] = _RealizedLineCurve(
                jnp.stack((plane_curve.origin[0], plane_source_parameter)),
                plane_curve.direction,
                parameter_domain=plane_curve.parameter_domain,
            )
        elif isinstance(plane_curve, ExactLineCurve):
            pcurves[seam.plane_coedge] = _RealizedLineCurve(
                jnp.stack((plane_curve.first[0], plane_source_parameter)),
                plane_curve.last - plane_curve.first,
                parameter_domain=plane_curve.parameter_domain,
            )
        else:
            raise ValueError(
                f"closed_seam_radius_map.plane_trim must be a native circle or source line; got {type(plane_curve).__name__}"
            )
        curved_curve = pcurves[seam.curved_coedge]
        pcurves[seam.curved_coedge] = _RealizedLineCurve(
            jnp.stack((jnp.zeros_like(source_parameter), source_parameter)),
            _angular_isoline_direction(curved_curve),
            parameter_domain=curved_curve.parameter_domain,
        )
        for coedge, value in ((blend_loop[0], lower), (blend_loop[2], upper)):
            line = pcurves[coedge]
            pcurves[coedge] = _RealizedLineCurve(
                jnp.stack((jnp.zeros_like(value), value)),
                _angular_isoline_direction(line),
                parameter_domain=line.parameter_domain,
            )
        for coedge in (blend_loop[1], blend_loop[3]):
            periodic = pcurves[coedge]
            if not isinstance(periodic, PeriodicPCurve):
                raise ValueError(
                    f"closed_seam_radius_map.seam must retain native phase; got {type(periodic).__name__}"
                )
            pcurves[coedge] = eqx.tree_at(lambda value: value.patch, periodic, patch)
        curves = list(geometry.curves)
        edge_uses = tuple(
            (face, coedge)
            for face, loops in enumerate(geometry.face_loops)
            for loop in loops
            for coedge in loop
        )
        affected = {
            seam.edge,
            seam.seam_edge,
            geometry.coedge_edges[seam.curved_coedge],
            geometry.coedge_edges[blend_loop[1]],
        }
        for edge in affected:
            face, coedge = next(
                (face, coedge)
                for face, coedge in edge_uses
                if geometry.coedge_edges[coedge] == edge
            )
            curves[geometry.edge_curves[edge]] = _RealizedTrimEdge(
                patches[face], pcurves[coedge]
            )
        ranges = geometry.edge_ranges.at[seam.seam_edge, self.endpoint].set(
            source_parameter
        )
        blend_edge = geometry.coedge_edges[blend_loop[1]]
        ranges = ranges.at[blend_edge].set(jnp.stack((lower, upper)))
        vertices = geometry.vertex_points
        for edge in (seam.edge, geometry.coedge_edges[seam.curved_coedge]):
            curve = curves[geometry.edge_curves[edge]]
            vertices = vertices.at[geometry.edge_vertices[edge][0]].set(
                curve.evaluate(jnp.asarray(0.0, dtype=jnp.float64))
            )
        return BRepParameterGeometry(
            patches=patches,
            source_geometry=geometry,
            parameter_bounds=bounds,
            orientation=model.orientation,
            trim_domains=model.trim_domains,
            triangle_parameters=None,
            source_patches=model.patches,
            vertex_points=vertices,
            curves=tuple(curves),
            pcurves=tuple(pcurves),
            edge_ranges=ranges,
        )


def _radius_limit(
    source: BRepModel,
    seam: _Seam,
    endpoint: int,
    other: float,
) -> float | None:
    """Outward lower bound of the first meridian/chart collision radius."""
    far_source = float(source.parameter_bounds[seam.curved_face, 1 - endpoint, 1])
    far = float(
        np.asarray(_canonical_profile_parameter(seam.curved_profile_affine, far_source))
    )
    q, d, z = seam.radial_sense, seam.axial_sense, seam.z
    sphere_pole = isinstance(seam.curved, SpherePatch) and far == -d * pi / 2 and q == -1
    if sphere_pole and other == 0:
        return None

    slope_bounds = _cone_slope_bounds(seam)

    def limits(point: Array) -> Array:
        # The dummy point keeps this a source-enclosure map, not a host formula.
        zero = point[0]
        values: list[Array] = []
        match seam.curved:
            case CylinderPatch():
                values.append(jnp.asarray(abs(far - z)) + zero)
                if q == 1:
                    values.append(seam.curved.radius + zero)
                if other > 0:
                    values.append(jnp.asarray(abs(seam.radial - other)) + zero)
            case ConePatch():
                slope = _cone_slope(seam) if slope_bounds is None else point[1]
                length = jnp.sqrt(1 + slope * slope)
                values.append((far - z) / (-d - q * slope / length) + zero)
                coefficient = -d * slope - q * length
                if q == 1:
                    values.append(-seam.radial / coefficient + zero)
                if other > 0:
                    values.append((other - seam.radial) / coefficient + zero)
            case SpherePatch():
                sphere = seam.curved.radius
                coefficient = 2 * (d * z - q * sphere)
                if q == 1:
                    values.append((z * z - sphere * sphere) / coefficient + zero)
                if not sphere_pole:
                    sine = jnp.sin(far)
                    values.append((z - sphere * sine) / (d - q * sine) + zero)
                if other > 0:
                    values.append(
                        (other * other - sphere * sphere + z * z) / coefficient + zero
                    )
        return jnp.stack(values)

    prepared = prepare_interval_function(
        limits, 1 if slope_bounds is None else 2, batch_capacity=1
    )
    if slope_bounds is None:
        zero_low = zero_high = np.asarray(((0.0,),), dtype=np.float64)
    else:
        slope_low, slope_high = slope_bounds
        zero_low = np.asarray(((0.0, slope_low),), dtype=np.float64)
        zero_high = np.asarray(((0.0, slope_high),), dtype=np.float64)
    low, _ = prepared.evaluate(zero_low, zero_high)
    upper = float(np.min(low))
    if not np.isfinite(upper) or upper <= 0:
        raise BRepBlendUnsupported(
            "closed seam radius map: a positive topology interval is uncertified"
        )
    return float(np.nextafter(upper, -np.inf))


def _chamfer_scale_limit(
    source: BRepModel,
    seam: _Seam,
    endpoint: int,
    other: float,
    setbacks: tuple[float, float],
    /,
) -> float:
    a, b = setbacks
    far_source = float(source.parameter_bounds[seam.curved_face, 1 - endpoint, 1])
    far = float(
        np.asarray(_canonical_profile_parameter(seam.curved_profile_affine, far_source))
    )
    q, d = seam.radial_sense, seam.axial_sense
    seam_parameter = float(
        np.asarray(
            _canonical_profile_parameter(seam.curved_profile_affine, seam.parameter)
        )
    )
    if isinstance(seam.curved, SpherePatch):
        raise BRepBlendUnsupported(
            "closed seam chamfer range: full-scale spherical generatrix sign is unproved"
        )

    slope_bounds = _cone_slope_bounds(seam)
    include_cone_limit = False
    if isinstance(seam.curved, ConePatch):
        if slope_bounds is None:
            include_cone_limit = d * float(_cone_slope(seam)) > 0
        else:
            slope_low, slope_high = slope_bounds
            if d * slope_low > 0:
                include_cone_limit = True
            elif d * slope_high > 0:
                raise BRepBlendUnsupported(
                    "closed seam chamfer range: cone slope sign is uncertified"
                )

    def limits(point: Array) -> Array:
        zero = point[0]
        if isinstance(seam.curved, ConePatch):
            slope = _cone_slope(seam) if slope_bounds is None else point[1]
            length = jnp.sqrt(1 + slope * slope)
        else:
            slope, length = jnp.asarray(0.0), jnp.asarray(1.0)
        values = [(far - seam_parameter) * length / (-d * b) + zero]
        if q == 1:
            values.append(seam.radial / a + zero)
        if other > 0:
            values.append(abs(seam.radial - other) / a + zero)
        if include_cone_limit:
            values.append(seam.radial * length / (d * b * slope) + zero)
        values.append(q * a - d * b * slope / length + zero)
        return jnp.stack(values)

    prepared = prepare_interval_function(
        limits, 1 if slope_bounds is None else 2, batch_capacity=1
    )
    if slope_bounds is None:
        zero_low = zero_high = np.asarray(((0.0,),), dtype=np.float64)
    else:
        slope_low, slope_high = slope_bounds
        zero_low = np.asarray(((0.0, slope_low),), dtype=np.float64)
        zero_high = np.asarray(((0.0, slope_high),), dtype=np.float64)
    low, high = prepared.evaluate(zero_low, zero_high)
    if low[0, -1] <= 0 <= high[0, -1]:
        raise BRepBlendUnsupported(
            "closed seam chamfer range: full-scale generatrix sign is uncertified"
        )
    upper = float(np.min(low[0, :-1]))
    if not np.isfinite(upper) or upper <= 0:
        raise BRepBlendUnsupported(
            "closed seam chamfer range: positive chart/axis scale interval is uncertified"
        )
    return float(np.nextafter(upper, -np.inf))


def closed_seam_parameter_range(
    source: BRepModel,
    edge_index: int,
    setbacks: tuple[float, float] | None,
    /,
) -> CertifiedParameterRange:
    """Reuse the interval-certified meridian/chart collision bound."""
    parameter = "radius" if setbacks is None else "setback_scale"
    try:
        seam = _seam(source, edge_index)
        endpoint = seam.curved_endpoint
        other = 0.0
        geometry = source.geometry
        for loop in geometry.face_loops[seam.plane_face]:
            if loop[0] != seam.plane_coedge:
                pcurve = geometry.pcurves[loop[0]]
                if not isinstance(pcurve, CircleCurve) or not np.array_equal(
                    pcurve.center, np.zeros(2)
                ):
                    raise BRepBlendUnsupported(
                        "closed seam blend: nonconcentric plane loops are unsupported"
                    )
                other = float(pcurve.radius)
        upper = (
            _radius_limit(source, seam, endpoint, other)
            if setbacks is None
            else _chamfer_scale_limit(source, seam, endpoint, other, setbacks)
        )
    except BRepBlendUnsupported as error:
        return unknown_parameter_range(parameter, str(error))
    if upper is None:
        return unknown_parameter_range(
            parameter, "closed seam meridian has no proved finite or unbounded full range"
        )
    return CertifiedParameterRange(
        parameter,
        (0.0, upper),
        end_status=("attained_exactly", "certified_inner_bound"),
        constraints=("parameter_domain", "trim_vanishes"),
        entity_ids=(
            (source.edge_ids[edge_index],),
            (source.face_ids[seam.plane_face], source.face_ids[seam.curved_face]),
        ),
        evidence=(
            "Native shared-frame circular seam with rectangular curved chart.",
            "Outward interval evaluation of the first meridian, radial, or chart collision; lower-rounded upper endpoint."
            if setbacks is None
            else f"Scale multiplies authored plane/curved setbacks {setbacks!r}; constant-sign generatrix and chart/radial bounds are interval-certified.",
        ),
    )


def prepare_closed_seam_radius_map(
    source: BRepModel, edge_index: int, radius: float, result: BRepModel
) -> AbstractBRepParameterMap:
    """Prepare the one radius parameter and every correlated native numerical leaf."""
    if result.tessellation is not None:
        raise BRepBlendUnsupported(
            "closed seam radius sensitivity cannot retain a tessellation fidelity certificate under parameter motion"
        )
    seam = _seam(source, edge_index)
    meridian = _fillet(seam, radius)
    _require_clearance(source, seam, meridian)
    endpoint = seam.curved_endpoint
    angle = atan2(
        meridian.curved_z - (seam.z - seam.axial_sense * radius),
        meridian.curved_radius - meridian.plane_radius,
    )
    other = 0.0
    geometry = source.geometry
    for loop in geometry.face_loops[seam.plane_face]:
        if loop[0] != seam.plane_coedge:
            pc = geometry.pcurves[loop[0]]
            if isinstance(pc, CircleCurve):
                other = float(pc.radius)
    radius_upper = _radius_limit(source, seam, endpoint, other)
    if radius_upper is not None and radius >= radius_upper:
        raise BRepBlendUnsupported(
            "closed seam radius map: radius is outside its certified open topology interval"
        )
    return _ClosedSeamRadiusMap(
        seam,
        jnp.asarray(radius, dtype=jnp.float64),
        result,
        source.source_revision,
        edge_index,
        meridian.blend_parameter - angle,
        endpoint,
        other,
        radius_upper,
    )
