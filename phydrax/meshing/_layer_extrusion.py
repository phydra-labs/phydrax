#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""CAD extrusion of planar walls into sweepable boundary-layer slabs.

A CAD_EXTRUSION control is realized before meshing: every planar wall face is
extruded along its inward normal by the schedule's total thickness, the source
is split by those prisms, and the published partition carries an
EXACT_SWEEP control whose caps are the extruded faces. Extrusion is accepted
only when each prism lies inside its controlled solid,
prisms of different walls are disjoint, and every slab's lateral faces lie on
the source boundary so the sweep never meets the core through a quad curtain.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._cad_revision import CADSelectionSet
from ..geometry._meshing_domain import MeshingDomain
from ..geometry._rational import rational_vector
from ..geometry.brep import BRepModel
from ..geometry.brep._boolean import boolean_brep
from ..geometry.brep._boolean_core import BRepBooleanPolicy
from ..geometry.brep._constructors import (
    brep_extrusion,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileArc,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
    ProfileSegment,
)
from ..geometry.brep._coordinate_request import coordinate_source_request
from ..geometry.brep._partition import (
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRole,
    native_arrangement,
    native_roundtrip_maps,
    NativePartition,
    publish_staged,
)
from ..geometry.brep._patches import CircleCurve, LineCurve, PlanePatch
from ..geometry.brep._revision import cad_revision_from_brep_model
from ..typing import checked
from ._controls import BoundaryLayerControl, BoundaryLayerRoute
from ._scope import MeshingEntityKind, MeshingScope
from ._trace import MeshingStageKind


_STAGE = MeshingStageKind.LAYER_GENERATION.value


import phydrax._external_resource as _owner_phydrax__external_resource
import phydrax.interchange._cad as _owner_phydrax_interchange__cad
import phydrax.interchange._cad_archive as _owner_phydrax_interchange__cad_archive
import phydrax.interchange._cad_brep_text as _owner_phydrax_interchange__cad_brep_text


def _unsupported(message: str, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.UNSUPPORTED_COMBINATION, message, stage=_STAGE
    )


def _plane(model: BRepModel, face: int, /) -> tuple[np.ndarray, float] | None:
    """Native outward unit normal and signed plane offset."""
    patch = model.patches[face]
    match patch:
        case PlanePatch():
            pass
        case _:
            return None
    normal = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError(
            "BRepModel.patches must provide a nonsingular plane chart for the selected extrusion wall; got an extrusion wall has a singular native plane chart"
        )
    orientation = float(np.asarray(model.orientation)[face])
    incident = model.topology.face_solids[face]
    if len(incident) == 1:
        owner = incident[0]
        position = model.topology.solid_faces[owner].index(face)
        orientation *= model.topology.solid_face_orientations[owner][position]
    normal *= orientation / norm
    return normal, float(normal @ np.asarray(patch.origin))


def _wall_profile(model: BRepModel, face: int, /) -> PlanarProfile:
    geometry = model.geometry
    patch = model.patches[face]
    if geometry is None:
        raise _unsupported(
            "BRepModel.geometry must provide declared native curves, p-curves, and oriented loops; got missing or incompatible declared native curves, p-curves, and oriented loops"
        )
    match patch:
        case PlanePatch():
            pass
        case _:
            raise _unsupported(
                "BRepModel.patches must provide planar CAD_EXTRUSION wall faces; got a nonplanar wall patch"
            )
    first_axis = np.asarray(patch.first_axis, dtype=np.float64)
    first_axis = first_axis / np.linalg.norm(first_axis)
    normal = np.cross(first_axis, np.asarray(patch.second_axis))
    normal /= np.linalg.norm(normal)
    second_axis = np.cross(normal, first_axis)
    origin = np.asarray(patch.origin)
    frame = np.stack((first_axis, second_axis))
    plane = ProfilePlane(
        (float(origin[0]), float(origin[1]), float(origin[2])),
        (float(first_axis[0]), float(first_axis[1]), float(first_axis[2])),
        (float(second_axis[0]), float(second_axis[1]), float(second_axis[2])),
    )
    loops = []
    ranges = np.asarray(geometry.edge_ranges)
    points = np.asarray(geometry.vertex_points)
    for loop in geometry.face_loops[face]:
        vertices = []
        segments: list[ProfileSegment] = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            sense = geometry.coedge_senses[coedge]
            start, end = geometry.edge_vertices[edge]
            point = points[start if sense > 0 else end]
            planar = frame @ (point - origin)
            vertices.append((float(planar[0]), float(planar[1])))
            curve_index = geometry.edge_curves[edge]
            if curve_index < 0:
                raise _unsupported(
                    "BRepModel.geometry must provide planar wall loops with supported noncollapsed line or circle trims; got a planar extrusion wall cannot contain a collapsed trim edge"
                )
            curve = geometry.curves[curve_index]
            match curve:
                case LineCurve() as curve:
                    segments.append(ProfileLine())
                case CircleCurve() as curve:
                    if ranges[edge, 1] - ranges[edge, 0] > 2.0 * np.pi:
                        raise _unsupported(
                            "BRepModel.geometry must provide planar wall loops with supported noncollapsed line or circle trims; got a planar wall arc cannot traverse its circle more than once"
                        )
                    center = frame @ (np.asarray(curve.center) - origin)
                    counterclockwise = (
                        sense
                        * float(
                            np.cross(
                                np.asarray(curve.first_axis),
                                np.asarray(curve.second_axis),
                            )
                            @ normal
                        )
                        > 0.0
                    )
                    segments.append(
                        ProfileArc((float(center[0]), float(center[1])), counterclockwise)
                    )
                case _:
                    raise _unsupported(
                        f"BRepModel.geometry must provide noncollapsed line or circle wall trims; got unsupported trim carrier {type(curve).__name__}"
                    )
        loops.append(ProfileLoop(tuple(vertices), tuple(segments)))
    return PlanarProfile(plane, loops[0], tuple(loops[1:]))


def _scope(
    model: BRepModel, dimension: int, identifiers: Iterable[int], /
) -> MeshingScope:
    domain = MeshingDomain.from_brep(model)
    definitions = (
        domain.region_source_indices
        if dimension == 3
        else domain.source_indices[dimension]
    )
    requested = set(identifiers)
    if not requested <= set(definitions):
        raise ValueError(
            "BoundaryLayerControl.wall_scope must belong to its native geometry inventory; got incompatible Extrusion scope definitions"
        )
    slots = domain.scope_indices(dimension)
    selected = tuple(
        slots[row]
        for row, definition in enumerate(definitions)
        if definition in requested
    )
    return MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(sorted(selected), dtype=np.int64),
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

    @checked
    def __init__(
        self,
        source: BRepModel,
        control: BoundaryLayerControl,
        layer_solid_ids: tuple[int, ...],
        core_solid_ids: tuple[int, ...],
        slab_volumes: ArrayLike,
        maximum_relative_volume_residual: float,
        /,
    ) -> None:
        if control.route is not BoundaryLayerRoute.EXACT_SWEEP:
            raise TypeError(
                "BoundaryLayerExtrusion.control must be an EXACT_SWEEP BoundaryLayerControl; got incompatible control"
            )
        volumes = np.asarray(slab_volumes, dtype=np.float64)
        self.source = source
        self.control = control
        self.layer_solid_ids = tuple(int(value) for value in layer_solid_ids)
        self.core_solid_ids = tuple(int(value) for value in core_solid_ids)
        _host_slab_volumes = np.asarray(volumes, dtype=np.float64)
        (self.slab_volumes,) = jax.device_put((_host_slab_volumes,))
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


def _definition_domain(model: BRepModel, /) -> MeshingDomain:
    """Admit precisely the physical graphs the definition arrangement realizes."""
    geometry = model.geometry
    if geometry is None:
        raise _unsupported(
            "BRepModel.geometry must provide declared qualified source incidence; got missing or incompatible declared qualified source incidence"
        )
    if geometry.occurrences:
        if sorted(occurrence.solid for occurrence in geometry.occurrences) != list(
            range(model.topology.num_solids)
        ):
            raise _unsupported(
                "BRepModel.geometry must retain a physical occurrence/definition bijection and qualified incidence; got native solid partition requires a physical occurrence-to-definition bijection; repeated or uninstantiated solid definitions require qualified physical realization"
            )
        if any(
            occurrence.rotation != ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            or occurrence.translation != (0.0, 0.0, 0.0)
            for occurrence in geometry.occurrences
        ):
            raise _unsupported(
                "BRepModel.geometry must use identity occurrence placements for native extrusion; got a placed occurrence requiring qualified physical realization"
            )
        paths_by_definition: dict[tuple[str, int], set[tuple[str, ...]]] = {}
        for record in geometry.qualified_entity_incidence(model.source_revision):
            member = record.member
            paths_by_definition.setdefault((member.kind, member.index), set()).add(
                member.occurrence_path
            )
        if any(len(paths) != 1 for paths in paths_by_definition.values()):
            raise _unsupported(
                "BRepModel.geometry must retain a physical occurrence/definition bijection and qualified incidence; got native solid partition cannot erase distinct authored entity incidences; qualified graph fragmentation is required"
            )
    return MeshingDomain.from_brep(model)


def _validated_request(
    model: BRepModel, control: BoundaryLayerControl, /
) -> tuple[tuple[int, ...], list[int], tuple[int, ...]]:
    if control.route is not BoundaryLayerRoute.CAD_EXTRUSION:
        raise ValueError(
            "prepare_boundary_layer_extrusion.control must declare geometry-bound CAD_EXTRUSION wall and volume scopes; got prepare_boundary_layer_extrusion realizes CAD_EXTRUSION controls"
        )
    volume_scope = control.volume_scope
    if volume_scope is None:
        raise ValueError(
            "prepare_boundary_layer_extrusion.control must provide a wall face scope, a volume scope, and no cap; got missing or incompatible a wall face scope, a volume scope, and no cap"
        )
    domain = _definition_domain(model)
    for scope, dimension in ((control.wall_scope, 2), (volume_scope, 3)):
        if (
            scope.source_id != domain.source_id
            or scope.source_revision != domain.source_revision
            or scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or scope.entity_dimension != dimension
            or scope.entity_set_id != domain.entity_set_id(dimension)
        ):
            raise ValueError(
                "prepare_boundary_layer_extrusion.control must bind this native qualified geometry inventory; got incompatible control scopes"
            )
    wall_rows = domain.resolve_indices(
        2, np.asarray(control.wall_scope.entity_ids, dtype=np.int64)
    )
    region_rows = domain.resolve_indices(
        3, np.asarray(volume_scope.entity_ids, dtype=np.int64)
    )
    selected_regions = set(int(row) for row in region_rows)
    walls = tuple(domain.source_indices[2][row] for row in wall_rows)
    owners = []
    adjacency = np.asarray(domain.patch_regions, dtype=np.int64)
    for row in wall_rows:
        incident = {int(region) for region in adjacency[row] if region >= 0}
        selected = incident & selected_regions
        if len(selected) != 1 or len(incident) != 1:
            raise _unsupported(
                "prepare_boundary_layer_extrusion.control must be a qualified boundary face of one controlled physical solid; got incompatible extruded wall"
            )
        owners.append(domain.region_source_indices[selected.pop()])
    solids = tuple(domain.region_source_indices[row] for row in region_rows)
    return walls, owners, solids


def _extrusion_prisms(
    model: BRepModel,
    walls: tuple[int, ...],
    total: float,
    tessellation: BRepTessellationPolicy,
    /,
) -> tuple[tuple[BRepModel, ...], tuple[tuple[np.ndarray, float], ...]]:
    prisms = []
    planes = []
    for wall in walls:
        plane = _plane(model, wall)
        if plane is None:
            raise _unsupported(
                "BRepModel.patches must provide planar CAD_EXTRUSION wall faces; got a nonplanar wall patch"
            )
        profile = _wall_profile(model, wall)
        normal, _ = plane
        prisms.append(
            brep_extrusion(
                profile,
                -total * normal,
                coordinate_contract=model.coordinate_contract,
                tessellation=tessellation,
                source_id=f"native-layer-slab:{model.source_revision}:face:{wall}",
            )
        )
        planes.append(plane)
    return tuple(prisms), tuple(planes)


def _partition_plan(
    source: BRepModel, prisms: tuple[BRepModel, ...], overwrite: bool, /
) -> BRepPartitionPlan:
    revision = cad_revision_from_brep_model(source)
    operands = tuple(
        BRepPartitionOperand(
            f"source-solid:{solid}",
            source,
            BRepPartitionRole.REGION,
            CADSelectionSet.from_revision(revision, (f"solid:{solid}",)),
        )
        for solid in range(source.topology.num_solids)
    ) + tuple(
        BRepPartitionOperand(f"layer:{index}", prism, BRepPartitionRole.REGION)
        for index, prism in enumerate(prisms)
    )
    return BRepPartitionPlan(
        source.coordinate_contract,
        operands,
        BRepPartitionPolicy(
            tuple(f"layer:{index}" for index in range(len(prisms)))
            + tuple(
                f"source-solid:{solid}" for solid in range(source.topology.num_solids)
            ),
            overwrite=overwrite,
        ),
    )


def _check_arrangement(
    plan: BRepPartitionPlan,
    owners: list[int],
    tessellation: BRepTessellationPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> NativePartition:
    policy = BRepBooleanPolicy(
        maximum_cells=plan.policy.maximum_cells,
        maximum_faces=plan.policy.maximum_faces,
        sewing=plan.policy.sewing,
        tessellation=tessellation,
    )
    operands = {operand.operand_id: operand for operand in plan.operands}
    controlled: dict[int, BRepModel] = {}
    slabs = tuple(operands[f"layer:{index}"].model for index in range(len(owners)))
    for index, owner in enumerate(owners):
        if owner not in controlled:
            operand = operands[f"source-solid:{owner}"]
            owner_plan = BRepPartitionPlan(
                plan.coordinate_contract,
                (operand,),
                BRepPartitionPolicy(
                    (operand.operand_id,),
                    maximum_cells=plan.policy.maximum_cells,
                    maximum_faces=plan.policy.maximum_faces,
                    sewing=plan.policy.sewing,
                ),
            )
            controlled[owner] = native_arrangement(
                owner_plan, policy, budget=budget
            ).model
        outside = boolean_brep(
            slabs[index],
            controlled[owner],
            "difference",
            policy=policy,
            budget=budget,
        )
        if not outside.empty:
            raise _unsupported(
                "BoundaryLayerControl.volume_scope must contain each extruded wall prism; got a prism that leaves its controlled solid"
            )
        for previous in slabs[:index]:
            overlap = boolean_brep(
                previous, slabs[index], "intersection", policy=policy, budget=budget
            )
            if not overlap.empty:
                raise _unsupported(
                    "BoundaryLayerControl.wall_scope must define disjoint extrusion slabs; got slabs of different walls overlap"
                )
    native = native_arrangement(plan, policy, budget=budget)
    return native


def _solid_volume(model: BRepModel, solid: int, /) -> float:
    """-coordinate divergence volume of planar native face loops."""
    geometry = model.geometry
    if geometry is None:
        raise _unsupported(
            "BRepModel.geometry must provide declared native topology; got missing or incompatible declared native topology"
        )
    points = tuple(rational_vector(point) for point in np.asarray(geometry.vertex_points))
    volume = Fraction()
    for face in model.topology.solid_faces[solid]:
        match model.patches[face]:
            case PlanePatch():
                pass
            case _:
                raise _unsupported(
                    "BRepModel.geometry must provide a native curved-face measure certificate; got missing or incompatible a native curved-face measure certificate"
                )
        shell_signs = tuple(
            sense
            for shell in geometry.solid_shells[solid]
            for candidate, sense in zip(
                geometry.shell_faces[shell],
                geometry.shell_orientations[shell],
                strict=True,
            )
            if candidate == face
        )
        if len(shell_signs) != 1:
            raise ValueError(
                "BRepModel.geometry must have exactly one oriented shell occurrence; got incompatible slab face"
            )
        orientation = shell_signs[0] * int(np.asarray(model.orientation)[face])
        for loop in geometry.face_loops[face]:
            vertices = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                start, end = geometry.edge_vertices[edge]
                vertices.append(
                    points[start if geometry.coedge_senses[coedge] > 0 else end]
                )
            first = vertices[0]
            for second, third in zip(vertices[1:-1], vertices[2:], strict=True):
                determinant = (
                    first[0] * (second[1] * third[2] - second[2] * third[1])
                    - first[1] * (second[0] * third[2] - second[2] * third[0])
                    + first[2] * (second[0] * third[1] - second[1] * third[0])
                )
                volume += orientation * determinant / 6
    return abs(float(volume))


def _classify_slabs(
    model: BRepModel,
    slab_ids: tuple[int, ...],
    planes: tuple[tuple[np.ndarray, float], ...],
    expected_volumes: np.ndarray,
    total: float,
    scale: float,
    /,
) -> tuple[list[int], list[int], np.ndarray, float]:
    tolerance = 1.0e-9 * scale
    bottoms, tops, volumes, residuals = [], [], [], []
    for slab, (normal, offset), expected in zip(
        slab_ids, planes, expected_volumes, strict=True
    ):
        volume = _solid_volume(model, slab)
        residual = abs(volume - float(expected)) / float(expected)
        if residual > 1.0e-9:
            raise _unsupported(
                "BoundaryLayerExtrusion.source must retain slab measure, wall/cap faces, and exterior lateral incidence; got the published native partition lost the extruded slab volume"
            )
        bottom, top = [], []
        for face in model.topology.solid_faces[slab]:
            plane = _plane(model, face)
            if plane is None or abs(abs(float(plane[0] @ normal)) - 1.0) > 1.0e-12:
                continue
            level = float(normal @ (plane[0] * plane[1]))
            if abs(level - offset) <= tolerance:
                bottom.append(face)
            elif abs(level - (offset - total)) <= tolerance:
                top.append(face)
        if len(bottom) != 1 or len(top) != 1:
            raise _unsupported(
                "BoundaryLayerExtrusion.source must retain slab measure, wall/cap faces, and exterior lateral incidence; got an extruded slab lacks one wall face and one cap face"
            )
        laterals = set(model.topology.solid_faces[slab]) - {bottom[0], top[0]}
        if any(len(model.topology.face_solids[face]) != 1 for face in laterals):
            raise _unsupported(
                "BoundaryLayerExtrusion.source must lie on the source boundary (full-width walls); got incompatible Extruded slab lateral faces"
            )
        bottoms.append(bottom[0])
        tops.append(top[0])
        volumes.append(volume)
        residuals.append(residual)
    return bottoms, tops, np.asarray(volumes, dtype=np.float64), max(residuals)


@checked
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
    """Partition wall-extrusion slabs and bind their EXACT_SWEEP control.

    The returned source is the published split BRep; mesh it with
    ``result.control`` (straight SWEEP fill) and region controls naming
    ``layer_solid_ids`` and ``core_solid_ids``.
    """

    if not isinstance(overwrite, bool):
        raise TypeError(
            "prepare_boundary_layer_extrusion.destination must be a bool; got incompatible overwrite"
        )
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in (".brep", ".brp", ".phx"):
        raise ValueError(
            "prepare_boundary_layer_extrusion.destination must provide .brep, .brp, or .phx; got missing or incompatible .brep, .brp, or .phx"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    walls, owners, controlled_solids = _validated_request(source, control)
    tessellation = BRepTessellationPolicy(
        linear_deflection=linear_deflection, angular_deflection=angular_deflection
    )
    total = control.schedule.total_thickness
    prisms, planes = _extrusion_prisms(source, walls, total, tessellation)
    plan = _partition_plan(source, prisms, overwrite)
    # arrangement occupancy decides containment and front collision;
    # sampled point location and centroid equality cannot establish either.
    with coordinate_source_request((plan,)) as budget:
        native = _check_arrangement(plan, owners, tessellation, budget=budget)
    expected_volumes = np.asarray(
        [_solid_volume(prism, 0) for prism in prisms], dtype=np.float64
    )
    native_slabs = []
    for index in range(len(prisms)):
        matches = tuple(
            solid
            for solid, owner in enumerate(native.solid_regions)
            if owner == f"layer:{index}"
        )
        if len(matches) != 1:
            raise _unsupported(
                "prepare_boundary_layer_extrusion.destination must remain one connected solid; got incompatible native extruded slab"
            )
        native_slabs.append(matches[0])
    scale = max(1.0, float(np.max(np.abs(np.asarray(source.mesh_vertices)), initial=0.0)))
    _classify_slabs(
        native.model, tuple(native_slabs), planes, expected_volumes, total, scale
    )
    controlled = set(controlled_solids)
    controlled_owners = {f"source-solid:{value}" for value in controlled}
    native_core = tuple(
        solid
        for solid, owner in enumerate(native.solid_regions)
        if owner in controlled_owners
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.extrusion-", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    staging = Path(name)
    staging.unlink()
    try:
        if target.suffix.lower() == ".phx":
            _owner_phydrax_interchange__cad_archive.save_brep_archive(
                native.model, staging
            )
            partitioned = _owner_phydrax_interchange__cad_archive.load_brep_archive(
                staging
            )
            exported_provenance = imported_provenance = None
        else:
            # External text is a new source revision; entities are mapped by
            # the codec's entity-reference provenance, not model identity.
            exported_provenance = (
                _owner_phydrax_interchange__cad_brep_text.write_brep_text(
                    native.model, staging
                ).provenance
            )
            decoded = _owner_phydrax_interchange__cad_brep_text.read_brep_text(
                staging,
                _owner_phydrax_interchange__cad.CadImportPolicy(
                    source.coordinate_contract,
                    _owner_phydrax__external_resource.ResourceLimits(
                        64 * 1024 * 1024, 128, 1_000_000, 10_000_000, 0
                    ),
                    tessellation=tessellation,
                ),
                trusted_root=target.parent,
                source_length_unit=source.coordinate_contract.length_unit,
            )
            restored = decoded.model
            imported_provenance = decoded.provenance
            partitioned = BRepModel(
                patches=restored.patches,
                parameter_bounds=restored.parameter_bounds,
                orientation=restored.orientation,
                trim_domains=restored.trim_domains,
                topology=restored.topology,
                coordinate_contract=restored.coordinate_contract,
                mesh_vertices=restored.mesh_vertices,
                mesh_faces=restored.mesh_faces,
                triangle_face_ids=restored.triangle_face_ids,
                triangle_parameters=restored.triangle_parameters,
                tessellation_deviation_bounds=restored.tessellation_deviation_bounds,
                tessellation_normal_bounds=restored.tessellation_normal_bounds,
                mesh_vertex_source_dimensions=restored.mesh_vertex_source_dimensions,
                mesh_vertex_source_indices=restored.mesh_vertex_source_indices,
                mesh_vertex_parameters=restored.mesh_vertex_parameters,
                mesh_chart_restriction_vertices=restored.mesh_chart_restriction_vertices,
                mesh_chart_restriction_edges=restored.mesh_chart_restriction_edges,
                mesh_chart_restriction_endpoint_parameters=(
                    restored.mesh_chart_restriction_endpoint_parameters
                ),
                mesh_chart_restriction_parameters=(
                    restored.mesh_chart_restriction_parameters
                ),
                coedge_deviation_bounds=restored.coedge_deviation_bounds,
                triangle_occurrence_ids=restored.triangle_occurrence_ids,
                vertex_occurrence_ids=restored.vertex_occurrence_ids,
                physical_tags=restored.physical_tags,
                report=replace(restored.report, source_id=str(target)),
                geometry=restored.geometry,
            )
        solid_map, _ = native_roundtrip_maps(
            native.model, partitioned, exported_provenance, imported_provenance
        )
        slabs = tuple(solid_map[solid] for solid in native_slabs)
        core = tuple(solid_map[solid] for solid in native_core)
        bottoms, tops, volumes, residual = _classify_slabs(
            partitioned, slabs, planes, expected_volumes, total, scale
        )
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
        order = np.argsort(np.asarray(slabs, dtype=np.int64))
        result = BoundaryLayerExtrusion(
            partitioned,
            sweep,
            tuple(slabs[index] for index in order),
            tuple(sorted(core)),
            volumes[order],
            residual,
        )
        publish_staged(staging, target, overwrite)
        return result
    finally:
        staging.unlink(missing_ok=True)


__all__ = ["BoundaryLayerExtrusion", "prepare_boundary_layer_extrusion"]
