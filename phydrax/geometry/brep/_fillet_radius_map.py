#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Correlated radius realization for rational planar-prism edge fillets."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..design._schema import (
    DesignState,
    ParameterBinding,
    ParameterCollector,
    ParameterId,
)
from ._coordinate_request import coordinate_source_request
from ._differentiable import (
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
)
from ._model import BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
)
from ._planar_blend import planar_blend_parameter_range


@final
class PlanarFilletRadiusMap(AbstractBRepParameterMap):
    """Bind one radius to its cylinder, adjacent planes, native edges and trims."""

    source: BRepModel
    edge_index: int = eqx.field(static=True)
    radius: Array

    def __init__(self, source: BRepModel, edge_index: int, radius: float, /) -> None:
        self.source = source
        self.edge_index = edge_index
        self.radius = jnp.asarray(radius, dtype=jnp.float64)

    def compile(
        self, context: ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        geometry = model.require_geometry()
        cylinders = tuple(
            index
            for index, patch in enumerate(model.patches)
            if isinstance(patch, CylinderPatch)
        )
        if len(cylinders) != 1:
            raise ValueError(
                "PlanarFilletRadiusMap.model must contain one blend cylinder; got a different cylinder count"
            )
        cylinder_face = cylinders[0]
        moving = frozenset(
            vertex
            for loop in geometry.face_loops[cylinder_face]
            for coedge in loop
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        )
        cylinder = model.patches[cylinder_face]
        if not isinstance(cylinder, CylinderPatch):
            raise RuntimeError(
                "PlanarFilletRadiusMap.cylinder must retain its admitted carrier; got a different patch type"
            )
        side_faces = tuple(
            face
            for face, patch in enumerate(model.patches)
            if isinstance(patch, PlanePatch)
            and float(np.asarray(patch.first_axis) @ np.asarray(cylinder.axis)) == 0.0
            and float(np.asarray(patch.second_axis) @ np.asarray(cylinder.axis)) != 0.0
        )
        mesh_owner = _radius_mesh_owner(model, cylinder_face, side_faces)
        # Prepare scientific bounds when the parameter map is compiled, not when
        # optional result evidence is requested. Reporting never changes schema.
        with coordinate_source_request((self.source, self.edge_index)) as budget:
            parameter_range = planar_blend_parameter_range(
                self.source, self.edge_index, None, budget=budget
            )
        binding = context.bind(
            ParameterId(
                f"{self.source.model_id}:edge:{self.edge_index}:fillet", "radius"
            ),
            self.radius,
            role="length",
            unit=context.coordinate_unit,
            physical_scale=float(np.asarray(self.radius)),
            bounds=(0.0, parameter_range.interval[1]),
        )
        return _BoundPlanarRadiusMap(
            binding=binding,
            model=model,
            reference_radius=self.radius,
            source_corner=self.source.require_geometry().vertex_points[
                self.source.require_geometry().edge_vertices[self.edge_index][0]
            ],
            cylinder_face=cylinder_face,
            moving_vertices=tuple(sorted(moving)),
            side_faces=side_faces,
            mesh_owner=mesh_owner,
        )


def _radius_mesh_owner(
    model: BRepModel,
    cylinder_face: int,
    side_faces: tuple[int, ...],
    /,
) -> tuple[int, ...]:
    """Assign moving mesh vertices with blend-cylinder ownership taking precedence."""
    owner = np.full((model.mesh_vertices.shape[0],), -1, dtype=np.int32)
    for face, vertices in zip(
        np.asarray(model.triangle_face_ids), np.asarray(model.mesh_faces), strict=True
    ):
        if face == cylinder_face or face in side_faces:
            for vertex in vertices:
                if owner[vertex] < 0 or face == cylinder_face:
                    owner[vertex] = face
    return tuple(owner.tolist())


@final
class _BoundPlanarRadiusMap(AbstractBoundBRepParameterMap):
    binding: ParameterBinding = eqx.field(static=True)
    model: BRepModel
    reference_radius: Array
    source_corner: Array
    cylinder_face: int = eqx.field(static=True)
    moving_vertices: tuple[int, ...] = eqx.field(static=True)
    side_faces: tuple[int, ...] = eqx.field(static=True)
    mesh_owner: tuple[int, ...] = eqx.field(static=True)

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        radius = self.binding.read(state)
        ratio = radius / self.reference_radius
        original = self.model.require_geometry()
        cylinder = self.model.patches[self.cylinder_face]
        if not isinstance(cylinder, CylinderPatch):
            raise RuntimeError(
                "PlanarFilletRadiusMap.cylinder must retain its prepared carrier; got a different patch type"
            )
        axis = cylinder.axis

        def shift(point: Array) -> Array:
            offset = point - self.source_corner
            radial = offset - jnp.sum(offset * axis, axis=-1, keepdims=True) * axis
            return point + (ratio - 1.0) * radial

        vertices = original.vertex_points
        moving = jnp.asarray(self.moving_vertices, dtype=jnp.int32)
        vertices = vertices.at[moving].set(shift(vertices[moving]))
        curves: list[AbstractCurve] = []
        ranges = original.edge_ranges
        for edge, source_index in enumerate(original.edge_curves):
            if source_index < 0:
                raise ValueError(
                    "PlanarFilletRadiusMap.edges must have nondegenerate carriers; got a pole edge"
                )
            curve = original.curves[source_index]
            first, last = original.edge_vertices[edge]
            match curve:
                case LineCurve():
                    start, end = vertices[first], vertices[last]
                    width = jnp.sum((end - start) * curve.direction) / jnp.sum(
                        curve.direction**2
                    )
                    ranges = ranges.at[edge].set(jnp.asarray((0.0, width)))
                    curves.append(eqx.tree_at(lambda value: value.origin, curve, start))
                case CircleCurve():
                    curves.append(
                        eqx.tree_at(
                            lambda value: (value.center, value.radius),
                            curve,
                            (shift(curve.center), radius),
                        )
                    )
                case _:
                    raise ValueError(
                        "PlanarFilletRadiusMap.carriers must be native lines or circles; got another curve family"
                    )
        patches = list(self.model.patches)
        patches[self.cylinder_face] = eqx.tree_at(
            lambda value: (value.origin, value.radius),
            cylinder,
            (shift(cylinder.origin), radius),
        )
        for face in self.side_faces:
            patch = patches[face]
            if not isinstance(patch, PlanePatch):
                raise RuntimeError(
                    "PlanarFilletRadiusMap.side must retain its plane carrier; got another patch family"
                )
            edge = original.coedge_edges[original.face_loops[face][0][0]]
            curve = curves[original.edge_curves[edge]]
            if not isinstance(curve, LineCurve):
                raise RuntimeError(
                    "PlanarFilletRadiusMap.side_curve must retain its generating line; got another curve family"
                )
            patches[face] = eqx.tree_at(lambda value: value.origin, patch, curve.origin)
        pcurves = list(original.pcurves)
        bounds = self.model.parameter_bounds
        for face, patch in enumerate(patches):
            if not isinstance(patch, PlanePatch):
                continue
            for loop in original.face_loops[face]:
                for coedge in loop:
                    edge = original.coedge_edges[coedge]
                    curve = curves[original.edge_curves[edge]]
                    first, last = original.edge_vertices[edge]

                    def chart(point: Array) -> Array:
                        difference = point - patch.origin
                        return jnp.stack(
                            (
                                jnp.sum(difference * patch.first_axis)
                                / jnp.sum(patch.first_axis**2),
                                jnp.sum(difference * patch.second_axis)
                                / jnp.sum(patch.second_axis**2),
                            ),
                            axis=-1,
                        )

                    match curve:
                        case LineCurve():
                            uv_first, uv_last = (
                                chart(vertices[first]),
                                chart(vertices[last]),
                            )
                            width = ranges[edge, 1] - ranges[edge, 0]
                            template = pcurves[coedge]
                            if not isinstance(template, LineCurve):
                                raise RuntimeError(
                                    "PlanarFilletRadiusMap.line_trim must retain its line carrier; got another trim family"
                                )
                            pcurves[coedge] = eqx.tree_at(
                                lambda value: (value.origin, value.direction),
                                template,
                                (uv_first, (uv_last - uv_first) / width),
                            )
                        case CircleCurve():
                            template = pcurves[coedge]
                            if not isinstance(template, CircleCurve):
                                raise RuntimeError(
                                    "PlanarFilletRadiusMap.circular_trim must retain its circle carrier; got another trim family"
                                )
                            pcurves[coedge] = eqx.tree_at(
                                lambda value: (
                                    value.center,
                                    value.first_axis,
                                    value.second_axis,
                                    value.radius,
                                ),
                                template,
                                (
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
                                            jnp.sum(
                                                curve.second_axis * patch.second_axis
                                            ),
                                        )
                                    ),
                                    radius,
                                ),
                            )
                        case _:
                            raise ValueError(
                                "PlanarFilletRadiusMap.carriers must be native lines or circles; got another curve family"
                            )
            if face in self.side_faces:
                edge = original.coedge_edges[original.face_loops[face][0][0]]
                bounds = bounds.at[face, 1, 0].set(ranges[edge, 1])
        geometry = eqx.tree_at(
            lambda value: (
                value.vertex_points,
                value.curves,
                value.edge_ranges,
                value.pcurves,
            ),
            original,
            (vertices, tuple(curves), ranges, tuple(pcurves)),
        )
        parameters = self._mesh_parameters(tuple(patches), bounds)
        return BRepParameterGeometry(
            patches=tuple(patches),
            geometry=geometry,
            parameter_bounds=bounds,
            triangle_parameters=parameters,
        )

    def _mesh_parameters(
        self, patches: tuple[AbstractSurfacePatch, ...], bounds: Array, /
    ) -> Array:
        old = self.model
        parameters = old.triangle_parameters
        transported = bounds[old.triangle_face_ids, 0, None] + (
            parameters - old.parameter_bounds[old.triangle_face_ids, 0, None]
        ) * (
            bounds[old.triangle_face_ids, 1, None]
            - bounds[old.triangle_face_ids, 0, None]
        ) / (
            old.parameter_bounds[old.triangle_face_ids, 1, None]
            - old.parameter_bounds[old.triangle_face_ids, 0, None]
        )
        points = old.mesh_vertices
        for face in (*self.side_faces, self.cylinder_face):
            corner_mask = old.triangle_face_ids == face
            corner_points = patches[face].evaluate(transported)
            selected = jnp.where(corner_mask[:, None, None], corner_points, 0.0)
            accumulated = (
                jnp.zeros_like(points)
                .at[old.mesh_faces.reshape(-1)]
                .add(selected.reshape((-1, 3)))
            )
            counts = (
                jnp.zeros((points.shape[0],), dtype=jnp.float64)
                .at[old.mesh_faces.reshape(-1)]
                .add(jnp.repeat(corner_mask, 3))
            )
            owners = jnp.asarray(self.mesh_owner, dtype=jnp.int32) == face
            points = jnp.where(
                owners[:, None], accumulated / jnp.maximum(counts[:, None], 1.0), points
            )
        for face, patch in enumerate(patches):
            if isinstance(patch, PlanePatch) and face not in self.side_faces:
                difference = points[old.mesh_faces] - patch.origin
                projected = jnp.stack(
                    (
                        jnp.sum(difference * patch.first_axis, axis=-1)
                        / jnp.sum(patch.first_axis**2),
                        jnp.sum(difference * patch.second_axis, axis=-1)
                        / jnp.sum(patch.second_axis**2),
                    ),
                    axis=-1,
                )
                transported = jnp.where(
                    (old.triangle_face_ids == face)[:, None, None], projected, transported
                )
        return transported
