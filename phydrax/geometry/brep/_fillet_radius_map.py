from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..design._schema import (
    DesignState,
    ParameterBinding,
    _ParameterCollector,
    ParameterId,
)
from ._differentiable import (
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
    _RealizedConicCurve,
    _RealizedCylinderPatch,
    _RealizedLineCurve,
    _RealizedPlanePatch,
)
from ._fillet import BRepBlendUnsupported
from ._model import BRepModel
from ._patches import (
    BRepCurve,
    CircleCurve,
    ConicRootLinePCurve,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
)
from ._planar_blend import planar_blend_parameter_range


@final
class PlanarFilletRadiusMap(AbstractBRepParameterMap):
    """Bind one radius to a perpendicular planar-prism fillet realization."""

    source: BRepModel
    reference: BRepModel
    edge_index: int = eqx.field(static=True)
    radius: Array

    def __init__(
        self,
        source: BRepModel,
        edge_index: int,
        radius: float,
        reference: BRepModel,
        /,
    ) -> None:
        if isinstance(edge_index, bool) or not isinstance(edge_index, int):
            raise TypeError("edge_index must be an integer.")
        if edge_index < 0 or edge_index >= len(source.edge_ids):
            raise ValueError("edge_index must reference a source edge.")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive.")
        geometry = reference.geometry
        supported_curves = (CircleCurve, ExactLineCurve, LineCurve)
        supported_pcurves = (*supported_curves, ConicRootLinePCurve)
        if any(index < 0 for index in geometry.edge_curves) or any(
            not isinstance(geometry.curves[index], supported_curves)
            for index in geometry.edge_curves
        ) or any(
            not isinstance(pcurve, supported_pcurves) for pcurve in geometry.pcurves
        ):
            raise BRepBlendUnsupported(
                "planar fillet radius sensitivity requires native line and circle carriers throughout the fixed topology"
            )
        self.source = source
        self.reference = reference
        self.edge_index = edge_index
        self.radius = jnp.asarray(radius, dtype=jnp.float64)

    def compile(
        self, context: _ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        if model.model_id != self.reference.model_id:
            raise ValueError("The radius map model must match its source-bound result.")
        if model.coordinate_contract.spatial_id != self.source.coordinate_contract.spatial_id:
            raise ValueError("The radius map source and result coordinate contracts differ.")
        if model.tessellation is not None:
            raise BRepBlendUnsupported(
                "planar fillet radius sensitivity cannot retain a tessellation fidelity certificate under parameter motion"
            )
        cylinders = tuple(
            index
            for index, patch in enumerate(model.patches)
            if isinstance(patch, CylinderPatch)
        )
        if len(cylinders) != 1:
            raise BRepBlendUnsupported(
                "planar fillet radius map requires one native blend cylinder"
            )
        cylinder_face = cylinders[0]
        geometry = model.geometry
        moving = frozenset(
            vertex
            for loop in geometry.face_loops[cylinder_face]
            for coedge in loop
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        )
        cylinder = model.patches[cylinder_face]
        if not isinstance(cylinder, CylinderPatch):
            raise RuntimeError("The admitted planar fillet cylinder changed carrier type.")
        side_faces = tuple(
            face
            for face, patch in enumerate(model.patches)
            if isinstance(patch, PlanePatch)
            and float(np.asarray(patch.first_axis) @ np.asarray(cylinder.axis)) == 0.0
            and float(np.asarray(patch.second_axis) @ np.asarray(cylinder.axis)) != 0.0
        )
        if not side_faces:
            raise BRepBlendUnsupported(
                "planar fillet radius map has no axial source side faces"
            )
        parameter_range = planar_blend_parameter_range(self.source, self.edge_index, None)
        upper = parameter_range.interval[1]
        if upper is None or upper <= 0.0:
            raise BRepBlendUnsupported(
                "planar fillet radius map lacks a positive certified radius interval"
            )
        radius = context.bind(
            ParameterId(
                f"{self.source.source_revision}:edge:{self.edge_index}:fillet", "radius"
            ),
            self.radius,
            role="length",
            physical_scale=float(np.asarray(self.radius)),
            bounds=(0.0, upper),
        )
        source_geometry = self.source.geometry
        corner = source_geometry.edge_vertices[self.edge_index][0]
        return _BoundPlanarRadiusMap(
            binding=radius,
            model=model,
            reference_radius=self.radius,
            source_corner=source_geometry.vertex_points[corner],
            cylinder_face=cylinder_face,
            moving_vertices=tuple(sorted(moving)),
            side_faces=side_faces,
        )


@final
class _BoundPlanarRadiusMap(AbstractBoundBRepParameterMap):
    binding: ParameterBinding = eqx.field(static=True)
    model: BRepModel
    reference_radius: Array
    source_corner: Array
    cylinder_face: int = eqx.field(static=True)
    moving_vertices: tuple[int, ...] = eqx.field(static=True)
    side_faces: tuple[int, ...] = eqx.field(static=True)

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        radius = self.binding.read(state)
        ratio = radius / self.reference_radius
        original = self.model.geometry
        cylinder = self.model.patches[self.cylinder_face]
        if not isinstance(cylinder, CylinderPatch):
            raise RuntimeError("The prepared planar fillet lost its cylinder carrier.")
        axis = cylinder.axis

        def shift(point: Array) -> Array:
            offset = point - self.source_corner
            radial = offset - jnp.sum(offset * axis, axis=-1, keepdims=True) * axis
            return point + (ratio - 1.0) * radial

        vertices = original.vertex_points
        moving = jnp.asarray(self.moving_vertices, dtype=jnp.int32)
        vertices = vertices.at[moving].set(shift(vertices[moving]))
        curves: list[BRepCurve] = []
        ranges = original.edge_ranges
        for edge, source_index in enumerate(original.edge_curves):
            if source_index < 0:
                raise ValueError("Planar fillet radius maps do not admit pole edges.")
            curve = original.curves[source_index]
            first, last = original.edge_vertices[edge]
            start, end = vertices[first], vertices[last]
            match curve:
                case LineCurve():
                    direction = end - start
                    ranges = ranges.at[edge].set(jnp.asarray((0.0, 1.0)))
                    curves.append(
                        _RealizedLineCurve(start, direction, parameter_domain=(0.0, 1.0))
                    )
                case ExactLineCurve():
                    direction = end - start
                    ranges = ranges.at[edge].set(jnp.asarray((0.0, 1.0)))
                    curves.append(
                        _RealizedLineCurve(start, direction, parameter_domain=(0.0, 1.0))
                    )
                case CircleCurve():
                    curves.append(
                        _RealizedConicCurve(
                            "circle",
                            shift(curve.center),
                            curve.first_axis,
                            curve.second_axis,
                            radius,
                            parameter_domain=curve.parameter_domain,
                            period=curve.period,
                            period_symbol=curve.native_period_symbol(),
                        )
                    )
                case _:
                    raise BRepBlendUnsupported(
                        "planar fillet radius map requires native line and circle edge carriers"
                    )
        patches = list(self.model.patches)
        patches[self.cylinder_face] = _RealizedCylinderPatch(
            shift(cylinder.origin),
            cylinder.first_axis,
            cylinder.second_axis,
            cylinder.axis,
            radius,
            periods=cylinder.periods,
            period_symbols=cylinder.native_period_symbols(),
        )
        for face in self.side_faces:
            patch = patches[face]
            if not isinstance(patch, PlanePatch):
                raise RuntimeError("A prepared axial side face lost its plane carrier.")
            edge = original.coedge_edges[original.face_loops[face][0][0]]
            curve = curves[original.edge_curves[edge]]
            if not isinstance(curve, _RealizedLineCurve):
                raise RuntimeError("A planar fillet side lost its generating line.")
            patches[face] = _RealizedPlanePatch(
                curve.origin, patch.first_axis, patch.second_axis
            )
        pcurves = list(original.pcurves)
        bounds = self.model.parameter_bounds
        for face, patch in enumerate(patches):
            if not isinstance(patch, (PlanePatch, _RealizedPlanePatch)):
                continue
            plane_origin = patch.origin
            plane_first_axis = patch.first_axis
            plane_second_axis = patch.second_axis
            for loop in original.face_loops[face]:
                for coedge in loop:
                    edge = original.coedge_edges[coedge]
                    curve = curves[original.edge_curves[edge]]
                    first, last = original.edge_vertices[edge]

                    def chart(point: Array) -> Array:
                        difference = point - plane_origin
                        return jnp.stack(
                            (
                                jnp.sum(difference * plane_first_axis)
                                / jnp.sum(plane_first_axis**2),
                                jnp.sum(difference * plane_second_axis)
                                / jnp.sum(plane_second_axis**2),
                            ),
                            axis=-1,
                        )

                    match curve:
                        case _RealizedLineCurve():
                            uv_first, uv_last = chart(vertices[first]), chart(vertices[last])
                            template = pcurves[coedge]
                            width = ranges[edge, 1] - ranges[edge, 0]
                            if not isinstance(template, (LineCurve, ExactLineCurve)):
                                raise BRepBlendUnsupported(
                                    "planar fillet line trim lost its affine chart carrier"
                                )
                            pcurves[coedge] = _RealizedLineCurve(
                                uv_first,
                                (uv_last - uv_first) / width,
                                parameter_domain=(0.0, 1.0),
                            )
                        case _RealizedConicCurve(kind="circle"):
                            template = pcurves[coedge]
                            if not isinstance(template, CircleCurve):
                                raise BRepBlendUnsupported(
                                    "planar fillet circular trim lost its native circle carrier"
                                )
                            pcurves[coedge] = _RealizedConicCurve(
                                "circle",
                                chart(curve.center),
                                jnp.stack(
                                    (
                                        jnp.sum(curve.first_axis * patch.first_axis),
                                        jnp.sum(curve.first_axis * patch.second_axis),
                                    )
                                ),
                                jnp.stack(
                                    (
                                        jnp.sum(curve.second_axis * patch.first_axis),
                                        jnp.sum(curve.second_axis * patch.second_axis),
                                    )
                                ),
                                radius,
                                parameter_domain=template.parameter_domain,
                                period=template.period,
                                period_symbol=template.native_period_symbol(),
                            )

                        case _:
                            raise BRepBlendUnsupported(
                                "planar fillet radius map requires native line and circle trim carriers"
                            )
            if face in self.side_faces:
                edge = original.coedge_edges[original.face_loops[face][0][0]]
                bounds = bounds.at[face, 1, 0].set(ranges[edge, 1])
        for face, patch in enumerate(patches):
            if not isinstance(patch, _RealizedCylinderPatch):
                continue
            for loop in original.face_loops[face]:
                for coedge in loop:
                    template = original.pcurves[coedge]
                    match template:
                        case LineCurve():
                            pcurves[coedge] = _RealizedLineCurve(
                                template.origin,
                                template.direction,
                                parameter_domain=template.parameter_domain,
                            )
                        case ExactLineCurve():
                            pcurves[coedge] = _RealizedLineCurve(
                                template.first,
                                template.last - template.first,
                                parameter_domain=template.parameter_domain,
                            )
                        case ConicRootLinePCurve():
                            pcurves[coedge] = _RealizedLineCurve(
                                jnp.stack((template.angle, template.height_origin)),
                                jnp.stack(
                                    (
                                        jnp.zeros_like(template.angle),
                                        template.height_direction,
                                    )
                                ),
                                parameter_domain=template.parameter_range,
                            )
                        case _:
                            raise BRepBlendUnsupported(
                                "planar fillet cylinder trims require native line carriers"
                            )
        return BRepParameterGeometry(
            patches=tuple(patches),
            source_geometry=original,
            parameter_bounds=bounds,
            orientation=self.model.orientation,
            trim_domains=self.model.trim_domains,
            triangle_parameters=None,
            source_patches=self.model.patches,
            vertex_points=vertices,
            curves=tuple(curves),
            pcurves=tuple(pcurves),
            edge_ranges=ranges,
        )
