#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Flat native geometry bundle of a CAD meshing domain.

Every surface, 3D curve and p-curve of a `MeshingDomain` becomes a node of
one geometry graph evaluated natively by ``phx_mc_cad_geometry_evaluate``:

- straight-line kinds (quadric surfaces, lines and conics, the analytic
  p-curves) are template programs generated offline from their JAX
  ``evaluate`` (``tools/generate_cad_geometry_programs.py``) and seeded here
  with the entity's binary64 coefficients;
- B-spline curves and surfaces (rational, by de Boor basis derivatives) and
  T-spline surfaces (rational Bernstein pieces of the realization's Bezier
  extraction) are native spline nodes, since knot-span and piece selection
  are not straight-line;
- placements, chart transports and period gauges are affine nodes; offset,
  ruled, extrusion, revolution, surface-mapped, isoparametric and trimmed
  kinds are composite nodes over their children;
- surface-surface intersection curves and their p-curves are implicit chart
  nodes: the source's fixed preconditioned chart iteration, differentiated by
  the implicit function theorem.

Offsets of quadrics and line chains reduce to the closed kind the source's
own evaluation branch describes. Nothing is approximated: a kind without a
native node is reported by `cad_bundle_support_issues`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import final

import numpy as np

from ..._meshcore import CadModel
from ...geometry._meshing_domain import (
    _PlacedSurfacePatch,
    MeshingDomain,
    PatchCurveUse,
)
from ...geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from ...geometry.brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
)
from ...geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
    TorusRootedCirclePCurve,
)
from ...geometry.brep._placed import PlacedCurve, PlacedSurface
from ...geometry.brep._tspline import TSplineSurfacePatch


@dataclass(frozen=True, slots=True)
class CadTemplate:
    """A straight-line source kind lowered offline to native bytecode.

    ``fields`` are the attribute paths of the entity's numerical leaves whose
    binary64 values seed the program, in seed order; ``static`` pins the
    static configuration the program was lowered for.
    """

    name: str
    kind: type[AbstractSurfacePatch] | type[AbstractCurve]
    fields: tuple[tuple[str, ...], ...]
    input_dimension: int
    output_dimension: int
    static: tuple[tuple[str, int], ...] = ()


_FRAME = (("first_axis",), ("second_axis",))


def _curve_templates(
    name: str,
    kind: type[AbstractCurve],
    fields: tuple[tuple[str, ...], ...],
    /,
) -> tuple[CadTemplate, ...]:
    return tuple(
        CadTemplate(f"{name}{dimension}", kind, fields, 1, dimension)
        for dimension in (3, 2)
    )


CAD_TEMPLATES: tuple[CadTemplate, ...] = (
    CadTemplate("plane", PlanePatch, (("origin",), *_FRAME), 2, 3),
    CadTemplate(
        "cylinder", CylinderPatch, (("origin",), *_FRAME, ("axis",), ("radius",)), 2, 3
    ),
    CadTemplate(
        "cone",
        ConePatch,
        (("origin",), *_FRAME, ("axis",), ("reference_radius",), ("semi_angle",)),
        2,
        3,
    ),
    CadTemplate(
        "sphere", SpherePatch, (("center",), *_FRAME, ("axis",), ("radius",)), 2, 3
    ),
    CadTemplate(
        "torus",
        TorusPatch,
        (("center",), *_FRAME, ("axis",), ("major_radius",), ("minor_radius",)),
        2,
        3,
    ),
    *_curve_templates("line", LineCurve, (("origin",), ("direction",))),
    *_curve_templates("exact_line", ExactLineCurve, (("first",), ("last",))),
    *_curve_templates("circle", CircleCurve, (("center",), *_FRAME, ("radius",))),
    *_curve_templates(
        "ellipse",
        EllipseCurve,
        (("center",), *_FRAME, ("first_radius",), ("second_radius",)),
    ),
    *_curve_templates(
        "parabola", ParabolaCurve, (("vertex",), *_FRAME, ("focal_length",))
    ),
    *_curve_templates(
        "hyperbola",
        HyperbolaCurve,
        (("center",), *_FRAME, ("first_radius",), ("second_radius",)),
    ),
    CadTemplate(
        "conic_root_line",
        ConicRootLinePCurve,
        (("angle",), ("height_origin",), ("height_direction",)),
        1,
        2,
    ),
    *(
        CadTemplate(
            f"torus_rooted_circle_{label}",
            TorusRootedCirclePCurve,
            (("longitude_offset",), ("root_line", "angle")),
            1,
            2,
            (("orientation", orientation),),
        )
        for label, orientation in (("forward", 1), ("reverse", -1))
    ),
    CadTemplate(
        "plane_cylinder_section",
        PlaneCylinderSectionPCurve,
        (("offset",), ("linear",), ("cosine",), ("sine",), ("angle_offset",)),
        1,
        2,
    ),
)

# Node kinds of the native geometry graph (PHX_MC_CAD_NODE_*).
_TEMPLATE = 0
_BSPLINE_CURVE = 1
_BSPLINE_SURFACE = 2
_TSPLINE_SURFACE = 3
_AFFINE = 4
_REPARAMETERIZED = 5
_OFFSET_SURFACE = 6
_OFFSET_CURVE = 7
_RULED = 8
_EXTRUSION = 9
_REVOLUTION = 10
_SURFACE_MAPPED = 11
_ISOPARAMETRIC = 12
_INTERSECTION = 13

# Highest native jet order (templates carry it through their Hessian program;
# a generic offset asks its base for one order more) and the order every
# entity table is read at.
_MAXIMUM_ORDER = 4
_ENTITY_ORDER = 2
_MAXIMUM_DEGREE = 31

type _Entity = AbstractSurfacePatch | AbstractCurve


def _leaf(entity: object, path: tuple[str, ...], /) -> object:
    for name in path:
        entity = getattr(entity, name)
    return entity


def _host(value: object, /) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _template_row(
    kind: type, dimension: int, entity: object, /
) -> tuple[int, CadTemplate] | None:
    for row, template in enumerate(CAD_TEMPLATES):
        if (
            template.kind is kind
            and template.output_dimension == dimension
            and all(getattr(entity, name) == value for name, value in template.static)
        ):
            return row, template
    return None


class CadBundleSupportError(ValueError):
    """A domain entity has no native geometry node."""


@final
class _NodeBank:
    """The geometry graph under construction; one node per source object."""

    def __init__(self) -> None:
        self.kinds: list[int] = []
        self.integers: list[np.ndarray] = []
        self.reals: list[np.ndarray] = []
        self.orders: list[int] = []
        self.children: list[tuple[int, ...]] = []
        self.issues: list[str] = []
        self._memo: dict[int, int] = {}
        self._entities: list[object] = []

    def _add(
        self,
        kind: int,
        integers: tuple[int, ...],
        reals: np.ndarray,
        children: tuple[int, ...],
        /,
    ) -> int:
        self.kinds.append(kind)
        self.integers.append(np.asarray(integers, dtype=np.int64))
        self.reals.append(np.ascontiguousarray(reals, dtype=np.float64).reshape((-1,)))
        self.children.append(children)
        self.orders.append(0)
        return len(self.kinds) - 1

    def node(self, entity: object, /) -> int:
        key = id(entity)
        if key in self._memo:
            return self._memo[key]
        handler = _HANDLERS.get(type(entity))
        if handler is None:
            raise CadBundleSupportError(
                f"no native geometry node for source kind {type(entity).__name__}"
            )
        index = handler(self, entity)
        # The memo holds the entity alive so its identity is never reused.
        self._entities.append(entity)
        self._memo[key] = index
        return index

    def template(self, entity: object, dimension: int, /) -> int:
        found = _template_row(type(entity), dimension, entity)
        if found is None:
            raise CadBundleSupportError(
                f"no template program for {type(entity).__name__} in dimension {dimension}"
            )
        row, template = found
        return self.template_from(row, template, entity)

    def template_from(self, row: int, template: CadTemplate, entity: object, /) -> int:
        coefficients = np.concatenate(
            [_host(_leaf(entity, path)).reshape((-1,)) for path in template.fields]
        )
        return self._add(_TEMPLATE, (row,), coefficients, ())

    def template_coefficients(
        self, name: str, coefficients: tuple[np.ndarray, ...], /
    ) -> int:
        row = next(
            index for index, template in enumerate(CAD_TEMPLATES) if template.name == name
        )
        return self._add(
            _TEMPLATE,
            (row,),
            np.concatenate([np.reshape(value, (-1,)) for value in coefficients]),
            (),
        )

    def affine(
        self, child: int, dimension: int, linear: np.ndarray, offset: np.ndarray, /
    ) -> int:
        return self._add(
            _AFFINE,
            (child, dimension),
            np.concatenate((linear.reshape((-1,)), offset.reshape((-1,)))),
            (child,),
        )

    def require_orders(self, roots: list[int], /) -> None:
        """Propagate the jet order each node is asked for from the entity tables."""
        order = list(self.orders)
        pending = [(root, _ENTITY_ORDER) for root in roots]
        while pending:
            node, requested = pending.pop()
            if requested <= order[node]:
                continue
            order[node] = requested
            extra = 1 if self.kinds[node] in (_OFFSET_SURFACE, _OFFSET_CURVE) else 0
            pending.extend((child, requested + extra) for child in self.children[node])
        for requested in order:
            if requested > _MAXIMUM_ORDER:
                self.issues.append(
                    f"generic offsets nested beyond order-{_MAXIMUM_ORDER} jets "
                    f"(order {requested} requested)"
                )
        self.orders = order


def _template_node(bank: _NodeBank, entity: object, /) -> int:
    dimension = (
        3
        if isinstance(entity, AbstractSurfacePatch)
        else entity.ambient_dimension
        if isinstance(entity, AbstractCurve)
        else 0
    )
    return bank.template(entity, dimension)


def _bspline_curve_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, BSplineCurve):
        raise TypeError("expected a BSplineCurve.")
    points = _host(curve.control_points)
    count, dimension = points.shape
    if curve.degree > _MAXIMUM_DEGREE:
        raise CadBundleSupportError(
            f"B-spline curve degree {curve.degree} exceeds {_MAXIMUM_DEGREE}"
        )
    return bank._add(
        _BSPLINE_CURVE,
        (curve.degree, count, dimension),
        np.concatenate((_host(curve.knots), points.reshape((-1,)), _host(curve.weights))),
        (),
    )


def _bspline_surface_node(bank: _NodeBank, surface: object, /) -> int:
    if not isinstance(surface, BSplineSurfacePatch):
        raise TypeError("expected a BSplineSurfacePatch.")
    points = _host(surface.control_points)
    if max(surface.u_degree, surface.v_degree) > _MAXIMUM_DEGREE:
        raise CadBundleSupportError(f"B-spline surface degree exceeds {_MAXIMUM_DEGREE}")
    return bank._add(
        _BSPLINE_SURFACE,
        (surface.u_degree, surface.v_degree, points.shape[0], points.shape[1]),
        np.concatenate(
            (
                _host(surface.u_knots),
                _host(surface.v_knots),
                points.reshape((-1,)),
                _host(surface.weights).reshape((-1,)),
            )
        ),
        (),
    )


def _tspline_surface_node(bank: _NodeBank, surface: object, /) -> int:
    if not isinstance(surface, TSplineSurfacePatch):
        raise TypeError("expected a TSplineSurfacePatch.")
    # The current geometry: the source bank carried through its exact
    # transfer chain, as the surface's own evaluation.
    surface.validate_source_bank()
    points, weights = surface.control_points, surface.weights
    for plan in surface.transfer_plans:
        points, weights = plan.apply_homogeneous_geometry(points, weights)
    realization = surface.realization
    rows = np.asarray(realization.piece_coefficient_rows, dtype=np.int64)
    widths = np.asarray(realization.piece_local_widths, dtype=np.int64)
    # Slots beyond a piece's local width are padding.
    padded = rows.copy()
    padded[np.arange(rows.shape[1])[None, :] >= widths[:, None]] = -1
    pieces, width = rows.shape
    return bank._add(
        _TSPLINE_SURFACE,
        (pieces, _host(points).shape[0], width, *padded.reshape((-1,)).tolist()),
        np.concatenate(
            (
                _host(realization.piece_bounds).reshape((-1,)),
                _host(realization.piece_extraction_operators).reshape((-1,)),
                _host(points).reshape((-1,)),
                _host(weights),
            )
        ),
        (),
    )


def _placed_patch_node(bank: _NodeBank, surface: object, /) -> int:
    if not isinstance(surface, _PlacedSurfacePatch):
        raise TypeError("expected a placed source patch.")
    child = bank.node(surface.source)
    return bank.affine(child, 3, _host(surface.linear), _host(surface.offset))


def _placed_node(bank: _NodeBank, entity: object, /) -> int:
    if not isinstance(entity, (PlacedSurface, PlacedCurve)):
        raise TypeError("expected a placed surface or curve.")
    child = bank.node(entity.definition)
    return bank.affine(child, 3, _host(entity.rotation), _host(entity.translation))


def _affine_pcurve_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, AffinePCurve):
        raise TypeError("expected an AffinePCurve.")
    child = bank.node(curve.curve)
    return bank.affine(child, 2, _host(curve.matrix_value), _host(curve.offset_value))


def _periodic_pcurve_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, PeriodicPCurve):
        raise TypeError("expected a PeriodicPCurve.")
    child = bank.node(curve.source_curve)
    return bank.affine(child, 2, np.eye(2, dtype=np.float64), _host(curve.offset_value))


def _unit_normal(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    normal = np.cross(first, second)
    return normal / np.linalg.norm(normal)


def _offset_surface_node(bank: _NodeBank, surface: object, /) -> int:
    """The source's own evaluation branches, closed kinds first."""
    if not isinstance(surface, OffsetSurface):
        raise TypeError("expected an OffsetSurface.")
    layers: list[OffsetSurface] = []
    base: AbstractSurfacePatch = surface
    while type(base) is OffsetSurface:
        layers.append(base)
        base = base.base
    if type(base) is PlanePatch:
        normal = _unit_normal(_host(base.first_axis), _host(base.second_axis))
        origin = _host(base.origin)
        for layer in reversed(layers):
            origin = origin + float(layer.distance) * (layer.frame_sign or 1) * normal
        return bank.template_coefficients(
            "plane", (origin, _host(base.first_axis), _host(base.second_axis))
        )
    inner = surface.base
    sign = surface.frame_sign
    distance = float(surface.distance)
    witness = surface.exact_sphere_revolution_witness()
    if isinstance(inner, SpherePatch) and sign != 0:
        return bank.template_coefficients(
            "sphere",
            (
                _host(inner.center),
                _host(inner.first_axis),
                _host(inner.second_axis),
                _host(inner.axis),
                np.asarray(float(inner.radius) + sign * distance),
            ),
        )
    if witness is not None and isinstance(inner, RevolutionSurface):
        profile = inner.curve
        if not isinstance(profile, CircleCurve):
            raise CadBundleSupportError("a spherical revolution offset lost its circle")
        circle = bank.template_coefficients(
            f"circle{profile.ambient_dimension}",
            (
                _host(profile.center),
                _host(profile.first_axis),
                _host(profile.second_axis),
                np.asarray(float(profile.radius) + sign * distance),
            ),
        )
        return bank._add(
            _REVOLUTION,
            (circle,),
            np.concatenate((_host(inner.axis_origin), _host(inner.axis_direction))),
            (circle,),
        )
    if isinstance(inner, CylinderPatch) and sign != 0:
        return bank.template_coefficients(
            "cylinder",
            (
                _host(inner.origin),
                _host(inner.first_axis),
                _host(inner.second_axis),
                _host(inner.axis),
                np.asarray(float(inner.radius) + sign * distance),
            ),
        )
    if isinstance(inner, TorusPatch) and sign != 0:
        return bank.template_coefficients(
            "torus",
            (
                _host(inner.center),
                _host(inner.first_axis),
                _host(inner.second_axis),
                _host(inner.axis),
                _host(inner.major_radius),
                np.asarray(float(inner.minor_radius) + sign * distance),
            ),
        )
    if isinstance(inner, ConePatch) and sign != 0:
        # The normal offset of a cone is the cone shifted along its axis by
        # -s d tan / sqrt(1 + tan^2) with reference radius r0 + s d / sqrt(..).
        slope = math.tan(float(inner.semi_angle))
        secant = math.sqrt(1.0 + slope * slope)
        axis = _host(inner.axis)
        return bank.template_coefficients(
            "cone",
            (
                _host(inner.origin) - (sign * distance * slope / secant) * axis,
                _host(inner.first_axis),
                _host(inner.second_axis),
                axis,
                np.asarray(float(inner.reference_radius) + sign * distance / secant),
                _host(inner.semi_angle),
            ),
        )
    child = bank.node(inner)
    return bank._add(
        _OFFSET_SURFACE,
        (child,),
        np.asarray((distance * (sign or 1),), dtype=np.float64),
        (child,),
    )


def _offset_curve_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, OffsetCurve):
        raise TypeError("expected an OffsetCurve.")
    layers: list[OffsetCurve] = []
    base: AbstractCurve = curve
    while type(base) is OffsetCurve:
        layers.append(base)
        base = base.base
    dimension = curve.ambient_dimension
    if type(base) is LineCurve or type(base) is ExactLineCurve:
        # A line offset by constant normals stays a line of the same kind.
        exact = isinstance(base, ExactLineCurve)
        if isinstance(base, ExactLineCurve):
            first, last = _host(base.first), _host(base.last)
            tangent, origin = last - first, first
        elif isinstance(base, LineCurve):
            tangent, origin = _host(base.direction), _host(base.origin)
            last = origin
        else:
            raise TypeError("expected a line curve.")
        shift = np.zeros(dimension, dtype=np.float64)
        for layer in reversed(layers):
            if dimension == 2:
                normal = np.asarray((tangent[1], -tangent[0]), dtype=np.float64)
            elif layer.direction is None:
                raise CadBundleSupportError("a space-curve offset requires a direction")
            else:
                normal = np.cross(tangent, _host(layer.direction))
            shift = shift + float(layer.distance) * normal / np.linalg.norm(normal)
        if exact:
            return bank.template_coefficients(
                f"exact_line{dimension}", (origin + shift, last + shift)
            )
        return bank.template_coefficients(f"line{dimension}", (origin + shift, tangent))
    direction = curve.direction
    if (dimension == 2) != (direction is None):
        raise CadBundleSupportError(
            "an offset curve needs a direction exactly in three dimensions"
        )
    child = bank.node(curve.base)
    reals = [float(curve.distance)]
    if direction is not None:
        reals.extend(_host(direction).tolist())
    return bank._add(
        _OFFSET_CURVE,
        (child, int(direction is not None)),
        np.asarray(reals, dtype=np.float64),
        (child,),
    )


def _ruled_node(bank: _NodeBank, surface: object, /) -> int:
    if not isinstance(surface, RuledSurface):
        raise TypeError("expected a RuledSurface.")
    first, second = bank.node(surface.first), bank.node(surface.second)
    return bank._add(_RULED, (first, second), np.zeros(0), (first, second))


def _extrusion_node(bank: _NodeBank, surface: object, /) -> int:
    if not isinstance(surface, ExtrusionSurface):
        raise TypeError("expected an ExtrusionSurface.")
    child = bank.node(surface.curve)
    return bank._add(_EXTRUSION, (child,), _host(surface.direction), (child,))


def _revolution_node(bank: _NodeBank, surface: object, /) -> int:
    if not isinstance(surface, RevolutionSurface):
        raise TypeError("expected a RevolutionSurface.")
    child = bank.node(surface.curve)
    return bank._add(
        _REVOLUTION,
        (child,),
        np.concatenate((_host(surface.axis_origin), _host(surface.axis_direction))),
        (child,),
    )


def _surface_mapped_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, SurfaceMappedCurve):
        raise TypeError("expected a SurfaceMappedCurve.")
    surface, pcurve = bank.node(curve.surface), bank.node(curve.pcurve)
    return bank._add(_SURFACE_MAPPED, (surface, pcurve), np.zeros(0), (surface, pcurve))


def _isoparametric_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, SurfaceIsoparametricCurve):
        raise TypeError("expected a SurfaceIsoparametricCurve.")
    surface = bank.node(curve.surface)
    return bank._add(
        _ISOPARAMETRIC,
        (surface, curve.fixed_axis),
        np.asarray((curve.fixed_value,), dtype=np.float64),
        (surface,),
    )


def _trim_segment_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, CurveTrimSegment):
        raise TypeError("expected a CurveTrimSegment.")
    child = bank.node(curve.curve)
    return bank._add(
        _REPARAMETERIZED,
        (child, int(curve.reversed)),
        np.asarray((curve.first, curve.last), dtype=np.float64),
        (child,),
    )


def _intersection_charts(
    bank: _NodeBank, curve: IntersectionCurve, /
) -> tuple[int, int, np.ndarray, np.ndarray]:
    first, second = bank.node(curve.first.patch), bank.node(curve.second.patch)
    axes = np.asarray(curve.chart_axes, dtype=np.int64)
    charts = np.concatenate(
        (
            _host(curve.chart_start),
            _host(curve.chart_end),
            _host(curve.chart_lower),
            _host(curve.chart_upper),
            _host(curve.preconditioners).reshape((axes.size, 9)),
        ),
        axis=1,
    )
    return first, second, axes, charts


def _intersection_curve_node(bank: _NodeBank, curve: object, /) -> int:
    if not isinstance(curve, IntersectionCurve):
        raise TypeError("expected an IntersectionCurve.")
    first, second, axes, charts = _intersection_charts(bank, curve)
    return bank._add(
        _INTERSECTION,
        (first, second, axes.size, 0, -1, -1, 0, *axes.tolist()),
        np.concatenate((np.zeros(2), charts.reshape((-1,)))),
        (first, second),
    )


def _intersection_pcurve_node(bank: _NodeBank, pcurve: object, /) -> int:
    """The p-curve's chart selection: one fixed chart, or the branch's chart
    by parameter with the terminal seam kept on the span's own last chart."""
    if not isinstance(pcurve, IntersectionPCurve):
        raise TypeError("expected an IntersectionPCurve.")
    curve = pcurve.curve
    first, second, axes, charts = _intersection_charts(bank, curve)
    chart = int(pcurve.first)
    fixed, terminal = -1, -1
    if pcurve.last <= chart + 1.0:
        fixed = chart
    else:
        last_chart = math.ceil(pcurve.last) - 1
        if pcurve.last == last_chart + 1 and pcurve.last < axes.size:
            terminal = last_chart
    output = 1 if pcurve.side == "first" else 2
    return bank._add(
        _INTERSECTION,
        (
            first,
            second,
            axes.size,
            output,
            fixed,
            terminal,
            int(pcurve.reversed),
            *axes.tolist(),
        ),
        np.concatenate(
            (
                np.asarray((pcurve.first + pcurve.last, pcurve.last), dtype=np.float64),
                charts.reshape((-1,)),
            )
        ),
        (first, second),
    )


_HANDLERS: dict[type, Callable[[_NodeBank, object], int]] = {
    **{template.kind: _template_node for template in CAD_TEMPLATES},
    BSplineCurve: _bspline_curve_node,
    BSplineSurfacePatch: _bspline_surface_node,
    TSplineSurfacePatch: _tspline_surface_node,
    _PlacedSurfacePatch: _placed_patch_node,
    PlacedSurface: _placed_node,
    PlacedCurve: _placed_node,
    AffinePCurve: _affine_pcurve_node,
    PeriodicPCurve: _periodic_pcurve_node,
    OffsetSurface: _offset_surface_node,
    OffsetCurve: _offset_curve_node,
    RuledSurface: _ruled_node,
    ExtrusionSurface: _extrusion_node,
    RevolutionSurface: _revolution_node,
    SurfaceMappedCurve: _surface_mapped_node,
    SurfaceIsoparametricCurve: _isoparametric_node,
    CurveTrimSegment: _trim_segment_node,
    IntersectionCurve: _intersection_curve_node,
    IntersectionPCurve: _intersection_pcurve_node,
}


@dataclass(frozen=True, slots=True)
class CadGeometryBundle:
    """Read-only native geometry of one meshing domain, keyed by its identity."""

    bundle_id: str
    model: CadModel
    edge_curve_rows: np.ndarray


def _occurrence_placements(
    domain: MeshingDomain, /
) -> dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]]:
    """Binary64 placement (linear, offset) of every assembly occurrence path."""
    authority = domain.brep_authority
    if authority is None:
        return {}
    placements: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    for occurrence in authority.geometry.assembly_graph.iter_occurrences():
        matrix = occurrence.placement.matrix
        placements[occurrence.path] = (
            np.asarray(
                [[float(matrix[row][column]) for column in range(3)] for row in range(3)],
                dtype=np.float64,
            ),
            np.asarray([float(matrix[row][3]) for row in range(3)], dtype=np.float64),
        )
    return placements


def _curve_nodes(
    bank: _NodeBank, domain: MeshingDomain, edge_curve_rows: np.ndarray, /
) -> list[int]:
    """Build the carrier node for each regular domain curve."""
    authority = domain.brep_authority
    placements = _occurrence_placements(domain)
    nodes: list[int] = []
    for curve, row in enumerate(edge_curve_rows):
        if row < 0:
            continue
        if authority is not None:
            geometry = authority.geometry
            edge = domain.source_indices[1][curve]
            if geometry.edge_curves[edge] < 0:
                raise ValueError("regular domain curve has no BRep curve geometry.")
            source = geometry.curves[geometry.edge_curves[edge]]
            node = bank.node(source)
            path = domain.source_occurrences[1][curve]
            if path:
                linear, offset = placements[path]
                node = bank.affine(node, 3, linear, offset)
            nodes.append(node)
            continue
        patch, loop, use = (int(value) for value in domain.curve_owners[curve])
        owner = domain.patches[patch].loops[loop][use]
        surface = bank.node(domain.patches[patch].surface)
        pcurve = bank.node(owner.pcurve)
        nodes.append(
            bank._add(_SURFACE_MAPPED, (surface, pcurve), np.zeros(0), (surface, pcurve))
        )
    return nodes


def _ragged(rows: list[np.ndarray], dtype: type, /) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([row.size for row in rows], out=offsets[1:])
    values = np.concatenate(rows).astype(dtype) if rows else np.zeros(0, dtype=dtype)
    return offsets, values


@dataclass(slots=True)
class _Tables:
    """Entity tables of a model under construction (row order = source order)."""

    corner_points: np.ndarray
    surfaces: list[AbstractSurfacePatch] = field(default_factory=list)
    surface_nodes: list[int] = field(default_factory=list)
    surface_orientations: list[int] = field(default_factory=list)
    curve_nodes: list[int] = field(default_factory=list)
    curve_corners: list[tuple[int, int]] = field(default_factory=list)
    curve_ranges: list[tuple[float, float]] = field(default_factory=list)
    loop_offsets: list[int] = field(default_factory=lambda: [0])
    loop_surfaces: list[int] = field(default_factory=list)
    use_kinds: list[int] = field(default_factory=list)
    use_entities: list[int] = field(default_factory=list)
    use_senses: list[int] = field(default_factory=list)
    use_ranges: list[tuple[float, float]] = field(default_factory=list)
    use_nodes: list[int] = field(default_factory=list)
    region_offsets: list[int] = field(default_factory=lambda: [0])
    region_surfaces: list[int] = field(default_factory=list)
    region_orientations: list[int] = field(default_factory=list)

    def model(self, bank: _NodeBank, /) -> CadModel:
        integer_offsets, integers = _ragged(bank.integers, np.int64)
        real_offsets, reals = _ragged(bank.reals, np.float64)
        model = CadModel(
            node_kinds=np.asarray(bank.kinds, dtype=np.int32),
            node_integer_offsets=integer_offsets,
            node_integers=integers,
            node_real_offsets=real_offsets,
            node_reals=reals,
            corner_points=np.ascontiguousarray(self.corner_points, dtype=np.float64),
            surface_nodes=np.asarray(self.surface_nodes, dtype=np.int32),
            surface_orientations=np.asarray(self.surface_orientations, dtype=np.int8),
            surface_periods=np.asarray(
                [
                    [0.0 if period is None else period for period in surface.periods]
                    for surface in self.surfaces
                ],
                dtype=np.float64,
            ).reshape((-1, 2)),
            surface_parameter_domains=np.asarray(
                [
                    ((-np.inf, np.inf), (-np.inf, np.inf))
                    if surface.parameter_domain is None
                    else surface.parameter_domain
                    for surface in self.surfaces
                ],
                dtype=np.float64,
            ).reshape((-1, 2, 2)),
            curve_nodes=np.asarray(self.curve_nodes, dtype=np.int32),
            curve_corners=np.asarray(self.curve_corners, dtype=np.int32).reshape((-1, 2)),
            curve_ranges=np.asarray(self.curve_ranges, dtype=np.float64).reshape((-1, 2)),
            loop_use_offsets=np.asarray(self.loop_offsets, dtype=np.int64),
            loop_surfaces=np.asarray(self.loop_surfaces, dtype=np.int32),
            use_kinds=np.asarray(self.use_kinds, dtype=np.int8),
            use_entities=np.asarray(self.use_entities, dtype=np.int32),
            use_senses=np.asarray(self.use_senses, dtype=np.int8),
            use_ranges=np.asarray(self.use_ranges, dtype=np.float64).reshape((-1, 2)),
            use_pcurve_nodes=np.asarray(self.use_nodes, dtype=np.int32),
            region_surface_offsets=np.asarray(self.region_offsets, dtype=np.int64),
            region_surfaces=np.asarray(self.region_surfaces, dtype=np.int32),
            region_orientations=np.asarray(self.region_orientations, dtype=np.int8),
        )
        for array in model:
            array.setflags(write=False)
        return model


def _issues(bank: _NodeBank, roots: list[int], issues: list[str], /) -> tuple[str, ...]:
    if not issues:
        bank.require_orders(roots)
        issues = bank.issues
    return tuple(dict.fromkeys(issues))


def _build(domain: MeshingDomain, /) -> tuple[CadModel, np.ndarray, tuple[str, ...]]:
    bank = _NodeBank()
    issues: list[str] = []

    edge_curve_rows = np.full(len(domain.curves), -1, dtype=np.int32)
    regular = np.flatnonzero(domain.curve_owners[:, 0] >= 0)
    edge_curve_rows[regular] = np.arange(regular.size, dtype=np.int32)
    edge_curve_rows.setflags(write=False)

    def guarded(entity: object, /) -> int:
        # Every unsupported entity is reported, not only the first.
        try:
            return bank.node(entity)
        except CadBundleSupportError as error:
            issues.append(str(error))
            return -1

    tables = _Tables(domain.corner_points)
    for patch in domain.patches:
        tables.surfaces.append(patch.surface)
        tables.surface_nodes.append(guarded(patch.surface))
        tables.surface_orientations.append(patch.orientation)
    try:
        tables.curve_nodes = _curve_nodes(bank, domain, edge_curve_rows)
    except CadBundleSupportError as error:
        issues.append(str(error))
    for curve_index in regular:
        curve = domain.curves[curve_index]
        patch, loop, use = domain.curve_owners[curve_index]
        owner = domain.patches[int(patch)].loops[int(loop)][int(use)]
        tables.curve_corners.append((curve.start, curve.end))
        tables.curve_ranges.append((owner.first, owner.last))
    for patch_index, patch in enumerate(domain.patches):
        for loop in patch.loops:
            for use in loop:
                is_curve = isinstance(use, PatchCurveUse)
                tables.use_kinds.append(0 if is_curve else 1)
                entity = int(edge_curve_rows[use.curve]) if is_curve else use.corner
                if entity < 0:
                    raise ValueError("curve use refers to a pole edge.")
                tables.use_entities.append(entity)
                tables.use_senses.append(use.sense)
                tables.use_ranges.append((use.first, use.last))
                tables.use_nodes.append(guarded(use.pcurve))
            tables.loop_surfaces.append(patch_index)
            tables.loop_offsets.append(len(tables.use_kinds))
    for region in domain.regions:
        for patch, orientation in region.boundary:
            tables.region_surfaces.append(patch)
            tables.region_orientations.append(orientation)
        tables.region_offsets.append(len(tables.region_surfaces))
    found = _issues(
        bank, [*tables.surface_nodes, *tables.curve_nodes, *tables.use_nodes], issues
    )
    if found:
        return (
            _Tables(np.zeros((0, 3), dtype=np.float64)).model(_NodeBank()),
            edge_curve_rows,
            found,
        )
    return tables.model(bank), edge_curve_rows, ()


def cad_entity_model(
    surfaces: tuple[AbstractSurfacePatch, ...],
    curves: tuple[AbstractCurve, ...],
    pcurves: tuple[tuple[AbstractCurve, float, float], ...],
    /,
) -> CadModel:
    """The native model of loose source entities, for evaluation only.

    Surfaces and 3D curves fill their tables in order; each ``(p-curve, first,
    last)`` is a use, in order, of one loop on the first surface (required
    when p-curves are given). There are no corners, regions or topology.
    """
    if pcurves and not surfaces:
        raise ValueError("p-curve uses need a carrier surface.")
    if any(curve.ambient_dimension != 3 for curve in curves) or any(
        pcurve.ambient_dimension != 2 for pcurve, _, _ in pcurves
    ):
        raise ValueError("curves must be three-dimensional and p-curves planar.")
    bank = _NodeBank()
    tables = _Tables(np.zeros((1, 3), dtype=np.float64))
    for surface in surfaces:
        tables.surfaces.append(surface)
        tables.surface_nodes.append(bank.node(surface))
        tables.surface_orientations.append(1)
    for curve in curves:
        tables.curve_nodes.append(bank.node(curve))
        tables.curve_corners.append((0, 0))
        tables.curve_ranges.append((0.0, 1.0))
    for pcurve, first, last in pcurves:
        tables.use_kinds.append(1)
        tables.use_entities.append(0)
        tables.use_senses.append(1)
        tables.use_ranges.append((first, last))
        tables.use_nodes.append(bank.node(pcurve))
    if pcurves:
        tables.loop_surfaces.append(0)
        tables.loop_offsets.append(len(pcurves))
    found = _issues(
        bank, [*tables.surface_nodes, *tables.curve_nodes, *tables.use_nodes], []
    )
    if found:
        raise CadBundleSupportError("; ".join(found))
    return tables.model(bank)


_CACHE_CAPACITY = 32
_CACHE: OrderedDict[str, tuple[CadModel, np.ndarray, tuple[str, ...]]] = OrderedDict()


def _cached(domain: MeshingDomain, /) -> tuple[CadModel, np.ndarray, tuple[str, ...]]:
    key = domain.domain_id
    entry = _CACHE.get(key)
    if entry is None:
        entry = _build(domain)
        _CACHE[key] = entry
        if len(_CACHE) > _CACHE_CAPACITY:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return entry


def cad_bundle_support_issues(domain: MeshingDomain, /) -> tuple[str, ...]:
    """Source kinds or compositions of ``domain`` without native geometry."""
    if not isinstance(domain, MeshingDomain):
        raise TypeError("domain must be a MeshingDomain.")
    return _cached(domain)[2]


def cad_geometry_bundle(domain: MeshingDomain, /) -> CadGeometryBundle:
    """The native geometry bundle of ``domain``; cached by its content identity."""
    if not isinstance(domain, MeshingDomain):
        raise TypeError("domain must be a MeshingDomain.")
    model, edge_curve_rows, issues = _cached(domain)
    if issues:
        raise CadBundleSupportError(
            "the domain has source kinds without native geometry: " + "; ".join(issues)
        )
    return CadGeometryBundle(domain.domain_id, model, edge_curve_rows)


__all__ = [
    "CAD_TEMPLATES",
    "CadBundleSupportError",
    "CadGeometryBundle",
    "CadTemplate",
    "cad_bundle_support_issues",
    "cad_entity_model",
    "cad_geometry_bundle",
]
