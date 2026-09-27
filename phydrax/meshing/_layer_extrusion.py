#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact CAD extrusion of planar walls into sweepable boundary-layer slabs.

A CAD_EXTRUSION control is realized before meshing: every planar wall face is
extruded along its inward normal by the schedule's total thickness, the source
is split by those exact prisms, and the published partition carries an
EXACT_SWEEP control whose caps are the extruded faces. Extrusion is accepted
only when the topology is exact: each prism lies inside its controlled solid,
prisms of different walls are disjoint, and every slab's lateral faces lie on
the source boundary so the sweep never meets the core through a quad curtain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from OCP.BOPAlgo import BOPAlgo_Splitter
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GeomAbs import GeomAbs_Plane
from OCP.gp import gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopAbs import (
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_SOLID,
)
from OCP.TopoDS import TopoDS

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry.brep import BRepModel, persist_occt_shape, read_occt_shape
from ..geometry.brep._occt import _explore_unique
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._controls import BoundaryLayerControl, BoundaryLayerRoute
from ._scope import MeshingEntityKind, MeshingScope
from ._trace import MeshingStageKind


_STAGE = MeshingStageKind.LAYER_GENERATION.value


def _unsupported(message: str, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.UNSUPPORTED_COMBINATION, message, stage=_STAGE
    )


def _volume(shape: Any, /) -> tuple[float, np.ndarray]:
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    center = properties.CentreOfMass()
    return properties.Mass(), np.asarray((center.X(), center.Y(), center.Z()))


def _plane(face: Any, /) -> tuple[np.ndarray, float] | None:
    """Outward unit normal and offset of a planar face, or ``None`` when curved."""
    adaptor = BRepAdaptor_Surface(face)
    if adaptor.GetType() != GeomAbs_Plane:
        return None
    plane = adaptor.Plane()
    axis = plane.Axis().Direction()
    origin = plane.Location()
    normal = np.asarray((axis.X(), axis.Y(), axis.Z()), dtype=np.float64)
    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal
    return normal, float(normal @ np.asarray((origin.X(), origin.Y(), origin.Z())))


def _scope(model: BRepModel, dimension: int, identifiers: Any, /) -> MeshingScope:
    revision = model.report.source_revision
    return MeshingScope(
        model.report.source_id,
        revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        f"{revision}:brep:{dimension}",
        np.asarray(sorted(identifiers), dtype=np.int64),
    )


class BoundaryLayerExtrusion(StrictModule, NonTrainableState):
    """Published slab partition and the EXACT_SWEEP control that meshes it."""

    source: BRepModel
    control: BoundaryLayerControl
    layer_solid_ids: tuple[int, ...] = eqx.field(static=True)
    core_solid_ids: tuple[int, ...] = eqx.field(static=True)
    slab_volumes: Array
    maximum_relative_volume_residual: float = eqx.field(static=True)
    extrusion_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: BRepModel,
        control: BoundaryLayerControl,
        layer_solid_ids: tuple[int, ...],
        core_solid_ids: tuple[int, ...],
        slab_volumes: Any,
        maximum_relative_volume_residual: float,
        /,
    ) -> None:
        if not isinstance(source, BRepModel):
            raise TypeError("source must be BRepModel.")
        if not isinstance(control, BoundaryLayerControl) or (
            control.route is not BoundaryLayerRoute.EXACT_SWEEP
        ):
            raise TypeError("control must be an EXACT_SWEEP BoundaryLayerControl.")
        volumes = np.asarray(slab_volumes, dtype=np.float64)
        self.source = source
        self.control = control
        self.layer_solid_ids = tuple(int(value) for value in layer_solid_ids)
        self.core_solid_ids = tuple(int(value) for value in core_solid_ids)
        self.slab_volumes = jnp.asarray(volumes)
        self.maximum_relative_volume_residual = float(maximum_relative_volume_residual)
        self.extrusion_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-extrusion",
                "source": source.model_id,
                "control": control.control_id,
                "layer_solids": self.layer_solid_ids,
                "core_solids": self.core_solid_ids,
                "volumes": array_tree_fingerprint(volumes),
            }
        )


def _validated_request(model: BRepModel, control: BoundaryLayerControl, /) -> Any:
    if not isinstance(model, BRepModel):
        raise TypeError("source must be BRepModel.")
    if not isinstance(control, BoundaryLayerControl):
        raise TypeError("control must be BoundaryLayerControl.")
    if control.route is not BoundaryLayerRoute.CAD_EXTRUSION:
        raise ValueError(
            "prepare_boundary_layer_extrusion realizes CAD_EXTRUSION controls."
        )
    revision = model.report.source_revision
    for scope, dimension in ((control.wall_scope, 2), (control.volume_scope, 3)):
        if (
            # ty: ignore[unresolved-attribute]
            scope.source_id != model.report.source_id
            # ty: ignore[unresolved-attribute]
            or scope.source_revision != revision
            # ty: ignore[unresolved-attribute]
            or scope.entity_set_id != f"{revision}:brep:{dimension}"
        ):
            raise ValueError("The control scopes must bind this BRep source revision.")
    walls = tuple(int(value) for value in np.asarray(control.wall_scope.entity_ids))
    # ty: ignore[unresolved-attribute]
    solids = {int(value) for value in np.asarray(control.volume_scope.entity_ids)}
    owners = []
    for wall in walls:
        incident = set(model.topology.face_solids[wall]) & solids
        if len(incident) != 1 or len(model.topology.face_solids[wall]) != 1:
            raise _unsupported(
                "Every extruded wall must be a boundary face of one controlled solid."
            )
        owners.append(incident.pop())
    return walls, owners


def _extrusion_prisms(shape: Any, walls: Any, owners: Any, total: float, /) -> Any:
    # ty: ignore[unresolved-attribute]
    faces = _explore_unique(shape, TopAbs_FACE, TopoDS.Face_s)
    # ty: ignore[unresolved-attribute]
    solids = _explore_unique(shape, TopAbs_SOLID, TopoDS.Solid_s)
    prisms = []
    planes = []
    for wall, owner in zip(walls, owners, strict=True):
        plane = _plane(faces[wall])
        if plane is None:
            raise _unsupported("CAD_EXTRUSION extrudes planar wall faces only.")
        normal, _ = plane
        inward = -normal * total
        prism = BRepPrimAPI_MakePrism(faces[wall], gp_Vec(*inward.tolist())).Shape()
        prism_volume, _ = _volume(prism)
        inside, _ = _volume(BRepAlgoAPI_Common(solids[owner], prism).Shape())
        if abs(inside - prism_volume) > 1.0e-9 * prism_volume:
            raise _unsupported(
                "The extruded wall prism leaves its solid; the extrusion topology is not exact."
            )
        prisms.append(prism)
        planes.append(plane)
    for first in range(len(prisms)):
        for second in range(first + 1, len(prisms)):
            shared, _ = _volume(BRepAlgoAPI_Common(prisms[first], prisms[second]).Shape())
            if shared > 1.0e-12 * _volume(prisms[first])[0]:
                raise _unsupported(
                    "Extruded slabs of different walls overlap; use ADVANCING layers."
                )
    return prisms, planes


def _split(shape: Any, prisms: Any, /) -> Any:
    splitter = BOPAlgo_Splitter()
    splitter.AddArgument(shape)
    for prism in prisms:
        splitter.AddTool(prism)
    splitter.Perform()
    if splitter.HasErrors():
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "OCCT failed to split the source by the extruded boundary-layer slabs.",
            stage=_STAGE,
        )
    return splitter.Shape()


def _classify_slabs(
    model: BRepModel, prisms: Any, planes: Any, total: float, scale: float, /
) -> Any:
    shape, _, _ = read_occt_shape(model.report.source_id)
    # ty: ignore[unresolved-attribute]
    faces = _explore_unique(shape, TopAbs_FACE, TopoDS.Face_s)
    # ty: ignore[unresolved-attribute]
    solids = _explore_unique(shape, TopAbs_SOLID, TopoDS.Solid_s)
    measures = [_volume(solid) for solid in solids]
    tolerance = 1.0e-9 * scale
    slabs, bottoms, tops, volumes, residuals = [], [], [], [], []
    for prism, (normal, offset) in zip(prisms, planes, strict=True):
        volume, center = _volume(prism)
        matches = [
            index
            for index, (candidate, centroid) in enumerate(measures)
            if abs(candidate - volume) <= 1.0e-9 * volume
            and np.linalg.norm(centroid - center) <= tolerance
        ]
        if len(matches) != 1:
            raise _unsupported(
                "The published partition lost the exact extruded slab solid."
            )
        slab = matches[0]
        bottom, top = [], []
        for face in model.topology.solid_faces[slab]:
            plane = _plane(faces[face])
            if plane is None or abs(abs(float(plane[0] @ normal)) - 1.0) > 1.0e-12:
                continue
            level = float(normal @ (plane[0] * plane[1]))
            if abs(level - offset) <= tolerance:
                bottom.append(face)
            elif abs(level - (offset - total)) <= tolerance:
                top.append(face)
        if len(bottom) != 1 or len(top) != 1:
            raise _unsupported(
                "An extruded slab lacks one exact wall face and one cap face."
            )
        laterals = set(model.topology.solid_faces[slab]) - {bottom[0], top[0]}
        if any(len(model.topology.face_solids[face]) != 1 for face in laterals):
            raise _unsupported(
                "Extruded slab lateral faces must lie on the source boundary (full-width walls)."
            )
        slabs.append(slab)
        bottoms.append(bottom[0])
        tops.append(top[0])
        volumes.append(measures[slab][0])
        residuals.append(abs(measures[slab][0] - volume) / volume)
    return slabs, bottoms, tops, np.asarray(volumes), max(residuals)


def prepare_boundary_layer_extrusion(
    source: BRepModel,
    control: BoundaryLayerControl,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    overwrite: bool = False,
) -> BoundaryLayerExtrusion:
    """Partition exact wall-extrusion slabs and bind their EXACT_SWEEP control.

    The returned source is the published split BRep; mesh it with
    ``result.control`` (straight SWEEP fill) and region controls naming
    ``layer_solid_ids`` and ``core_solid_ids``.
    """
    walls, owners = _validated_request(source, control)
    shape, source_format, digest = read_occt_shape(source.report.source_id)
    if (
        source_format != source.report.source_format
        or digest != source.report.source_digest
    ):
        raise ValueError("The extrusion source bytes changed after import.")
    total = control.schedule.total_thickness
    prisms, planes = _extrusion_prisms(shape, walls, owners, total)
    partitioned = persist_occt_shape(
        _split(shape, prisms),
        destination,
        coordinate_contract=source.coordinate_contract,
        overwrite=overwrite,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    scale = max(1.0, float(np.max(np.abs(np.asarray(source.mesh_vertices)), initial=0.0)))
    slabs, bottoms, tops, volumes, residual = _classify_slabs(
        partitioned, prisms, planes, total, scale
    )
    # ty: ignore[unresolved-attribute]
    controlled = {int(value) for value in np.asarray(control.volume_scope.entity_ids)}
    # ty: ignore[unresolved-attribute]
    original = _explore_unique(shape, TopAbs_SOLID, TopoDS.Solid_s)
    published = _explore_unique(
        read_occt_shape(partitioned.report.source_id)[0],
        TopAbs_SOLID,
        # ty: ignore[unresolved-attribute]
        TopoDS.Solid_s,
    )
    core = []
    for index, solid in enumerate(published):
        if index in slabs:
            continue
        _, center = _volume(solid)
        inside = any(
            _volume(BRepAlgoAPI_Common(original[owner], solid).Shape())[0]
            > (1.0 - 1.0e-9) * _volume(solid)[0]
            for owner in controlled
        )
        del center
        if inside:
            core.append(index)
    sweep = BoundaryLayerControl(
        _scope(partitioned, 2, bottoms),
        control.schedule,
        route=BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=_scope(partitioned, 3, slabs),
        cap_scope=_scope(partitioned, 2, tops),
        corner=control.corner,
        feature_angle=control.feature_angle,
        minimum_thickness_fraction=control.minimum_thickness_fraction,
        growth_rate_bounds=control.growth_rate_bounds,
        maximum_corner_stretch=control.maximum_corner_stretch,
        smoothing_iterations=control.smoothing_iterations,
        core_maximum_size=control.core_maximum_size,
    )
    return BoundaryLayerExtrusion(
        partitioned, sweep, tuple(sorted(slabs)), tuple(sorted(core)), volumes, residual
    )


__all__ = ["BoundaryLayerExtrusion", "prepare_boundary_layer_extrusion"]
