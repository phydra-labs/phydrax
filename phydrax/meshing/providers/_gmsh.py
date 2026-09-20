#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module, util
from itertools import pairwise
from pathlib import Path
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Shape

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellBlock, CellGeometrySpec, CellMesh, lagrange_element
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._reference_cell import reference_cell_topology
from ...geometry.brep import (
    BRepEntityId,
    BRepModel,
    BRepPartitionResult,
    BRepSource,
    PlanarEmbedding,
)
from ...geometry.brep._occt import _explore_unique, read_occt_shape
from ...geometry.simplicial import TriangleMesh
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from ...logging import emit
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import audit_cell_mesh
from .._canonical import canonicalize_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceDescriptor,
    MeshingSourceKind,
    ProviderSupportReport,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import SweptLayerControl
from .._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionRole,
)
from .._planar_bands import PlanarBandResult
from .._quality import evaluate_cell_quality, evaluate_swept_layer_quality
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._session import AbstractMeshingSession, MeshingExecutionPolicy
from .._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    resolve_size_controls,
    SizeCombinationPolicy,
    SizeControlStrength,
    SizeFieldDomain,
    UniformSizeControl,
)
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


_GMSH_LOCK = threading.Lock()
_BRepMeshingSource = BRepModel | BRepSource | BRepPartitionResult | PlanarBandResult


def _brep_model(source: _BRepMeshingSource, /) -> BRepModel:
    if isinstance(source, PlanarBandResult):
        return source.partition.model
    if isinstance(source, BRepPartitionResult):
        return source.model
    if isinstance(source, BRepSource):
        return source.model
    if isinstance(source, BRepModel):
        return source
    raise TypeError(
        "source must be a BRepModel, BRepSource, BRepPartitionResult, or PlanarBandResult."
    )


def _scope_indices(
    source: BRepModel, scope: MeshingScope, dimension: int, /
) -> np.ndarray | None:
    counts = {
        0: source.report.num_vertices,
        1: source.report.num_edges,
        2: source.report.num_faces,
        3: source.topology.num_solids,
    }
    count = counts[dimension]
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    if (
        scope.source_id != source.report.source_id
        or scope.source_revision != source.report.source_revision
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != dimension
        or scope.entity_set_id != f"{source.report.source_revision}:brep:{dimension}"
        or np.any(identifiers >= count)
    ):
        return None
    return identifiers


def _layer_preflight_issues(
    source: BRepModel,
    controls: tuple[SweptLayerControl, ...],
    requested_kinds: set[str],
    geometry_order: int,
    /,
) -> list[str]:
    if not controls:
        return []
    issues = []
    if geometry_order != 1:
        issues.append("Gmsh swept layers support affine order-one geometry only")
    controlled: dict[int, tuple[int, SweptLayerControl]] = {}
    face_roles: dict[tuple[int, int], str] = {}
    for control_index, control in enumerate(controls):
        volume_ids = _scope_indices(source, control.volume_scope, 3)
        source_faces = _scope_indices(source, control.source_scope, 2)
        target_faces = _scope_indices(source, control.target_scope, 2)
        if volume_ids is None or not volume_ids.size:
            issues.append("SweptLayerControl volume scope is not a nonempty solid scope")
            continue
        if source_faces is None or target_faces is None:
            issues.append("SweptLayerControl caps are not face scopes of this BRep")
            continue
        volumes = {int(value) for value in volume_ids}
        if volumes & set(controlled):
            issues.append("SweptLayerControl volume scopes must be disjoint")
        for solid in volumes:
            controlled.setdefault(solid, (control_index, control))
        for role, faces in (("source", source_faces), ("target", target_faces)):
            assigned = set()
            for face_value in faces:
                face = int(face_value)
                owners = set(source.topology.face_solids[face])
                selected = owners & volumes
                if len(selected) != 1:
                    issues.append(
                        "Every swept cap face must bound exactly one controlled solid"
                    )
                    continue
                solid = selected.pop()
                if solid in assigned:
                    issues.append(
                        "Each controlled solid requires exactly one source and one target cap"
                    )
                assigned.add(solid)
                face_roles[solid, face] = role
            if assigned != volumes:
                issues.append(
                    "Swept cap scopes must cover every controlled solid exactly once"
                )
    all_solids = set(range(source.topology.num_solids))
    selected_solids = set(controlled)
    expected_kinds = (
        {"prism"} if selected_solids == all_solids else {"prism", "tetrahedron"}
    )
    if requested_kinds != expected_kinds:
        issues.append(
            "Gmsh swept output requires exactly "
            + (
                "prism cells for complete volume coverage"
                if expected_kinds == {"prism"}
                else "prism and tetrahedron cells for a swept slab with an unswept remainder"
            )
        )
    swept_tet_caps = set()
    for solid, (_, control) in controlled.items():
        for face in source.topology.solid_faces[solid]:
            role = face_roles.get((solid, face), "lateral")
            other_controlled = {
                owner for owner in source.topology.face_solids[face] if owner != solid
            } & selected_solids
            other_unswept = {
                owner for owner in source.topology.face_solids[face] if owner != solid
            } - selected_solids
            if role != "lateral" and other_unswept:
                swept_tet_caps.add(face)
            if role == "lateral" and other_unswept:
                issues.append(
                    "A swept lateral face would create a quad curtain adjoining an unswept volume"
                )
            if role != "lateral" and other_controlled:
                issues.append(
                    "Controlled solids may meet one another only through matching lateral sweep faces"
                )
            for other in other_controlled:
                _, other_control = controlled[other]
                if (
                    face_roles.get((other, face), "lateral") != "lateral"
                    or control.schedule.schedule_id != other_control.schedule.schedule_id
                ):
                    issues.append(
                        "Adjacent swept solids require matching schedules on their shared lateral face"
                    )
    if selected_solids != all_solids and not swept_tet_caps:
        issues.append(
            "A partial swept slab must meet the tetrahedral remainder through a source or target cap"
        )
    return list(dict.fromkeys(issues))


def _uniform_control_conflicts(
    controls: tuple[UniformSizeControl, ...],
    combination: SizeCombinationPolicy,
    /,
) -> bool:
    entity_sets = {control.scope.entity_set_id for control in controls}
    for entity_set_id in entity_sets:
        bound = tuple(
            control
            for control in controls
            if control.scope.entity_set_id == entity_set_id
        )
        entity_ids = np.unique(
            np.concatenate(
                tuple(
                    np.asarray(control.scope.entity_ids, dtype=np.int64)
                    for control in bound
                )
            )
        )
        for entity_id in entity_ids:
            active = tuple(
                control
                for control in bound
                if np.any(np.asarray(control.scope.entity_ids) == entity_id)
            )
            hard = tuple(
                control
                for control in active
                if control.strength is SizeControlStrength.HARD
            )
            lower = max(
                (
                    control.minimum_size
                    for control in hard
                    if control.minimum_size is not None
                ),
                default=0.0,
            )
            upper = min(
                (
                    control.maximum_size
                    for control in hard
                    if control.maximum_size is not None
                ),
                default=np.inf,
            )
            if lower > upper:
                return True
            pool = hard if hard else active
            if combination is SizeCombinationPolicy.REJECT_HARD_CONFLICTS and hard:
                choices = tuple(
                    np.clip(control.target_size, lower, upper) for control in hard
                )
            elif combination is SizeCombinationPolicy.EXPLICIT_PRIORITY:
                priority = max(control.priority for control in pool)
                choices = tuple(
                    np.clip(control.target_size, lower, upper)
                    for control in pool
                    if control.priority == priority
                )
            else:
                choices = ()
            if choices and any(value != choices[0] for value in choices):
                return True
    return False


def _semantic_surface_issues(
    source: BRepModel, specification: SurfaceMeshingSpec, /
) -> list[str]:
    issues = []
    if not specification.region_controls:
        return ["Every planar source face requires an explicit RegionControl"]
    occupied: set[int] = set()
    face_regions = np.empty((source.report.num_faces,), dtype=object)
    face_regions[:] = None
    for control in specification.region_controls:
        face_ids = _scope_indices(source, control.scope, 2)
        if face_ids is None:
            issues.append(
                f"RegionControl {control.region_name!r} is not a face scope of this BRep"
            )
            continue
        selected = {int(value) for value in face_ids}
        if occupied & selected:
            issues.append("RegionControl face scopes must be disjoint")
        occupied.update(selected)
        face_regions[face_ids] = control.region_name
        if not control.meshing_enabled or control.role is RegionRole.VOID:
            issues.append("Disabled and void RegionControl values are unsupported")
    expected_faces = set(range(source.report.num_faces))
    if occupied != expected_faces:
        issues.append("RegionControl scopes must exhaustively cover all source faces")
        return issues

    declared: set[int] = set()
    for control in specification.patch_controls:
        edge_ids = _scope_indices(source, control.scope, 1)
        if edge_ids is None:
            issues.append(
                f"PatchControl {control.name!r} is not an edge scope of this BRep"
            )
            continue
        matched = False
        for edge_value in edge_ids:
            edge = int(edge_value)
            owners = source.topology.edge_faces[edge]
            actual = tuple(sorted(str(face_regions[owner]) for owner in owners))
            if (
                len(owners) not in (1, 2)
                or len(set(actual)) != len(actual)
                or actual != control.adjacent_region_names
            ):
                issues.append(
                    f"PatchControl {control.name!r} does not match source edge adjacency"
                )
                continue
            declared.add(edge)
            matched = True
        if control.required and not matched:
            issues.append(
                f"Required PatchControl {control.name!r} has no matching source edge"
            )
    expected_patches = {
        edge
        for edge, owners in enumerate(source.topology.edge_faces)
        if len(owners) == 1
        or (len(owners) == 2 and face_regions[owners[0]] != face_regions[owners[1]])
    }
    missing = expected_patches - declared
    if missing:
        issues.append(
            f"Source BRep contains undeclared exterior/interface edges {sorted(missing)}"
        )
    return issues


class GmshOptions(StrictModule, NonTrainableState):
    algorithm_2d: int = eqx.field(static=True)
    algorithm_3d: int = eqx.field(static=True)
    terminal_output: bool = eqx.field(static=True)
    association_tolerance_factor: float = eqx.field(static=True)
    options_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        algorithm_2d: int = 6,
        algorithm_3d: int = 1,
        terminal_output: bool = False,
        association_tolerance_factor: float = 4.0,
    ):
        factor = float(association_tolerance_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("association_tolerance_factor must be positive and finite.")
        self.algorithm_2d = int(algorithm_2d)
        self.algorithm_3d = int(algorithm_3d)
        self.terminal_output = bool(terminal_output)
        self.association_tolerance_factor = factor
        self.options_id = canonical_fingerprint(
            {
                "kind": "gmsh-options",
                "algorithm_2d": int(algorithm_2d),
                "algorithm_3d": int(algorithm_3d),
                "terminal_output": bool(terminal_output),
                "association_tolerance_factor": factor,
            }
        )


class GmshMeshingPlan(StrictModule, NonTrainableState):
    source: BRepModel
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    options: GmshOptions
    support: ProviderSupportReport
    planar_bands: PlanarBandResult | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: BRepModel | BRepSource | BRepPartitionResult | PlanarBandResult,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        options: GmshOptions,
        support: ProviderSupportReport,
        /,
    ):
        model = _brep_model(source)
        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError("specification must be surface or volume meshing.")
        if not isinstance(options, GmshOptions):
            raise TypeError("options must be GmshOptions.")
        if not isinstance(support, ProviderSupportReport):
            raise TypeError("support must be ProviderSupportReport.")
        support.require_supported()
        planar_bands = source if isinstance(source, PlanarBandResult) else None
        self.source = model
        self.specification = specification
        self.options = options
        self.support = support
        self.planar_bands = planar_bands
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gmsh-meshing-plan",
                "source_revision": model.report.source_revision,
                "specification": specification.specification_id,
                "options": options.options_id,
                "support": support.report_id,
                "planar_bands": (
                    None if planar_bands is None else planar_bands.result_id
                ),
            }
        )

    def execute(self, /) -> CellMeshingResult:
        return GmshProvider(self.options).execute(self)


class GmshSession(AbstractMeshingSession):
    def __init__(
        self,
        provider: GmshProvider,
        policy: MeshingExecutionPolicy,
        /,
    ):
        if policy.execution_mode is not MeshingExecutionMode.IN_PROCESS:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Gmsh currently supports in-process execution only.",
            )
        if util.find_spec("gmsh") is None:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "The optional gmsh Python package is unavailable.",
            )
        if not _GMSH_LOCK.acquire(blocking=False):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Another in-process Gmsh session owns the global provider state.",
            )
        try:
            gmsh = import_module("gmsh")
            if gmsh.isInitialized():
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    "An external owner already initialized the global Gmsh session.",
                )
            gmsh.initialize()
        except BaseException:
            _GMSH_LOCK.release()
            raise
        self._provider = provider
        self._policy = policy
        self._gmsh = gmsh
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def version(self) -> str:
        return str(self._gmsh.__version__)

    def execute(self, plan: GmshMeshingPlan, /) -> CellMeshingResult:
        if self.closed:
            raise RuntimeError("Cannot execute with a closed Gmsh session.")
        if not isinstance(plan, GmshMeshingPlan):
            raise TypeError("plan must be GmshMeshingPlan.")
        return _execute_gmsh(self._gmsh, plan, self.version)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._gmsh.finalize()
        finally:
            self._closed = True
            _GMSH_LOCK.release()


def _validate_gmsh_semantics(
    source,
    specification,
    model,
    descriptor,
    target,
    scope,
    requested: set[str],
    layers,
    unsupported: list[str],
    /,
) -> bool:
    semantic_volume = False
    if isinstance(specification, VolumeMeshingSpec):
        semantic_volume = bool(
            model.topology.num_solids > 1
            or specification.region_controls
            or specification.patch_controls
        )
        if not descriptor.closed:
            unsupported.append(
                "Gmsh volume meshing requires at least one closed BRep solid"
            )
        if (
            isinstance(source, BRepModel)
            and not isinstance(source, BRepSource)
            and not semantic_volume
            and (layers or requested != {"tetrahedron"})
        ):
            unsupported.append(
                "Non-semantic BRepModel volume meshing supports tetrahedra only"
            )
        if semantic_volume and specification.periodic_constraints:
            unsupported.append(
                "Strict semantic volume meshing does not implement periodicity"
            )
        if layers:
            if specification.fill_strategy is not VolumeFillStrategy.SWEEP:
                unsupported.append("Gmsh layers require explicit straight SWEEP fill")
            if specification.periodic_constraints:
                unsupported.append(
                    "Swept layers cannot be combined with periodic constraints"
                )
            unsupported.extend(
                _layer_preflight_issues(
                    model,
                    layers,
                    requested,
                    target.geometry_order,
                )
            )
        elif specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
            unsupported.append(
                "Non-simplex Gmsh volume fill requires an explicit straight-sweep layer control"
            )
        if specification.region_seeds or specification.hole_seeds:
            unsupported.append(
                "Strict BRep volume meshing does not implement region or hole seeds"
            )
        if semantic_volume:
            if not specification.region_controls:
                unsupported.append("Every solid requires an explicit RegionControl")
            occupied: set[int] = set()
            region_names: set[str] = set()
            solid_region: dict[int, str] = {}
            for control in specification.region_controls:
                identifiers = _scope_indices(model, control.scope, 3)
                if identifiers is None:
                    unsupported.append(
                        "RegionControl scope is not a solid scope of this BRep"
                    )
                    continue
                selected = {int(value) for value in identifiers}
                if occupied & selected:
                    unsupported.append("RegionControl solid scopes must be disjoint")
                occupied.update(selected)
                for solid in selected:
                    solid_region.setdefault(solid, control.region_name)
                if control.region_name in region_names:
                    unsupported.append("RegionControl region names must be unique")
                region_names.add(control.region_name)
                if not control.meshing_enabled or control.role is RegionRole.VOID:
                    unsupported.append(
                        "Disabled and void RegionControl values are unsupported"
                    )
            if occupied != set(range(model.topology.num_solids)):
                unsupported.append(
                    "RegionControl scopes must exhaustively cover all source solids"
                )
            if occupied == set(range(model.topology.num_solids)):
                observed_internal_faces = {
                    face
                    for face, owners in enumerate(model.topology.face_solids)
                    if len(owners) == 2
                    and solid_region[owners[0]] != solid_region[owners[1]]
                }
                declared_internal_faces: set[int] = set()
                for control in specification.patch_controls:
                    face_ids = _scope_indices(model, control.scope, 2)
                    if face_ids is None:
                        unsupported.append(
                            f"PatchControl {control.name!r} is not a face scope of this BRep"
                        )
                        continue
                    matched = False
                    for face in face_ids:
                        owners = model.topology.face_solids[int(face)]
                        actual = tuple(sorted(solid_region[owner] for owner in owners))
                        if (
                            len(owners) not in (1, 2)
                            or len(set(actual)) != len(actual)
                            or actual != control.adjacent_region_names
                        ):
                            unsupported.append(
                                f"PatchControl {control.name!r} does not match source face adjacency"
                            )
                            continue
                        matched = True
                        if len(owners) == 2:
                            declared_internal_faces.add(int(face))
                    if control.required and not matched:
                        unsupported.append(
                            f"Required PatchControl {control.name!r} has no matching source face"
                        )
                unexpected = observed_internal_faces - declared_internal_faces
                if unexpected:
                    unsupported.append(
                        f"Source BRep contains undeclared inter-region faces {sorted(unexpected)}"
                    )
            uniform = tuple(
                control
                for control in specification.size_controls
                if isinstance(control, UniformSizeControl)
            )
            if _uniform_control_conflicts(uniform, specification.size_combination):
                unsupported.append(
                    "Overlapping UniformSizeControl values have incompatible hard intervals or targets"
                )
    else:
        semantic_surface = bool(
            target.ambient_dimension == 2
            or specification.region_controls
            or specification.patch_controls
            or isinstance(source, PlanarBandResult)
        )
        if semantic_surface:
            unsupported.extend(_semantic_surface_issues(model, specification))
        elif model.topology.num_solids > 1:
            unsupported.append(
                "Multi-solid BRep surface meshing is outside the strict semantic volume path"
            )
        if isinstance(source, PlanarBandResult):
            if (
                specification.planar_embedding is None
                or specification.planar_embedding.embedding_id
                != source.embedding.embedding_id
            ):
                unsupported.append(
                    "Planar band source and SurfaceMeshingSpec embeddings must match exactly"
                )
            if source.partition.model.source_revision != model.source_revision:
                unsupported.append(
                    "Planar band source revision does not match its partition model"
                )
    return semantic_volume


def _validate_gmsh_periodicity(specification, model, unsupported: list[str], /) -> None:
    for constraint in specification.periodic_constraints:
        scopes = (constraint.source_scope, constraint.target_scope)
        if any(
            value.entity_dimension not in (1, 2)
            or _scope_indices(model, value, value.entity_dimension) is None
            for value in scopes
        ):
            unsupported.append(
                "Gmsh periodic scopes must select source BRep curves or surfaces"
            )
        if scopes[0].entity_dimension != scopes[1].entity_dimension or len(
            scopes[0].entity_ids
        ) != len(scopes[1].entity_ids):
            unsupported.append(
                "Gmsh periodic source/target scopes must have equal dimension and cardinality"
            )
        if np.asarray(constraint.transform).shape != (4, 4):
            unsupported.append(
                "Gmsh periodic transforms must be 4-by-4 in source coordinates"
            )


def _validate_gmsh_size_controls(
    specification,
    model,
    scope,
    semantic_volume: bool,
    unsupported: list[str],
    /,
) -> None:
    covered_solids: set[int] = set()
    for control in specification.size_controls:
        if isinstance(control, ProximitySizeControl):
            unsupported.append("Gmsh local proximity sizing has not been lowered")
            continue
        if isinstance(control, CurvatureSizeControl):
            if control.scope.scope_id != scope.scope_id:
                unsupported.append("Gmsh local curvature sizing has not been lowered")
            if control.use_faceted_curvature:
                unsupported.append(
                    "Gmsh curvature sizing uses CAD curvature, not faceted curvature"
                )
            if semantic_volume:
                unsupported.append(
                    "Semantic volume sizing supports solid-scoped UniformSizeControl values only"
                )
            continue
        if not isinstance(control, UniformSizeControl):
            unsupported.append("Gmsh size control is unsupported")
            continue
        local_dimension = control.scope.entity_dimension
        local_ids = (
            _scope_indices(model, control.scope, local_dimension)
            if local_dimension in (0, 1, 2, 3)
            else None
        )
        if local_ids is None:
            unsupported.append(
                "UniformSizeControl scope is not an entity scope of this BRep"
            )
            continue
        if semantic_volume:
            if local_dimension != 3:
                unsupported.append(
                    "Semantic volume sizing supports UniformSizeControl on source solids only"
                )
            else:
                covered_solids.update(int(value) for value in local_ids)
        elif control.scope.scope_id != scope.scope_id and (
            control.minimum_size is not None
            or control.maximum_size is not None
            or control.maximum_growth_rate is not None
        ):
            unsupported.append(
                "Local UniformSizeControl bounds and growth are not yet audited"
            )
    uniform = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if _uniform_control_conflicts(uniform, specification.size_combination):
        unsupported.append(
            "Overlapping UniformSizeControl values have incompatible hard intervals or targets"
        )
    whole_hard = tuple(
        control
        for control in specification.size_controls
        if not isinstance(control, ProximitySizeControl)
        and control.scope.scope_id == scope.scope_id
        and control.strength is SizeControlStrength.HARD
    )
    hard_minimum = max(
        (
            control.minimum_size
            for control in whole_hard
            if control.minimum_size is not None
        ),
        default=0.0,
    )
    hard_maximum = min(
        (
            control.maximum_size
            for control in whole_hard
            if control.maximum_size is not None
        ),
        default=np.inf,
    )
    if hard_minimum > hard_maximum:
        unsupported.append("Whole-source hard size intervals conflict")
    if (
        specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY
        and not semantic_volume
        and any(
            isinstance(control, UniformSizeControl)
            and control.scope.scope_id != scope.scope_id
            for control in specification.size_controls
        )
    ):
        unsupported.append(
            "Local Gmsh size fields do not yet lower explicit-priority overlaps"
        )
    if (
        specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY
        and any(
            isinstance(control, CurvatureSizeControl)
            for control in specification.size_controls
        )
        and len(specification.size_controls) > 1
    ):
        unsupported.append(
            "Gmsh does not lower explicit priority across curvature and other controls"
        )
    if semantic_volume and covered_solids != set(range(model.topology.num_solids)):
        unsupported.append(
            "Solid-scoped UniformSizeControl values must cover every source solid"
        )


def _audit_gmsh_mesh(
    specification,
    mesh,
    geometry,
    boundary,
    attributes,
    associations,
    zones,
    patches,
    region_zones,
    cell_solid_ids,
    family_policy,
    requested_kinds,
    minimum_jacobian,
    semantic_surface,
    semantic_volume,
    periodic_requested,
    periodic_achieved,
    layer_audit,
    layer_interface_achieved,
    band_requested,
    band_achieved,
    size_field_ids,
    /,
):
    quality_evaluation = evaluate_cell_quality(mesh, mesh.coordinates)
    audit = audit_cell_mesh(
        mesh,
        geometry,
        quality_evaluation,
        patches=patches,
        associations=associations,
        attributes=attributes,
        zones=zones,
        boundary=boundary,
    )
    if not audit.passed:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=audit.quality.worst_cell_global_ids,
        )
    achieved_kinds = {block.cell_kind for block in mesh.blocks}
    connectivity = mesh.connectivity
    if not isinstance(
        connectivity,
        (
            PolygonalConnectivity,
            TetrahedralConnectivity,
            HexahedralConnectivity,
            PolyhedralConnectivity,
        ),
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh surface/volume conversion requires two- or three-dimensional connectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    connectivity_edges = np.asarray(connectivity.edges, dtype=np.int32)
    edge_lengths = np.linalg.norm(
        np.asarray(mesh.coordinates)[connectivity_edges[:, 1]]
        - np.asarray(mesh.coordinates)[connectivity_edges[:, 0]],
        axis=1,
    )
    minimum_edge = float(np.min(edge_lengths))
    maximum_edge = float(np.max(edge_lengths))
    vertex_minimum = np.full((mesh.coordinates.shape[0],), np.inf)
    vertex_maximum = np.zeros((mesh.coordinates.shape[0],), dtype=np.float64)
    np.minimum.at(vertex_minimum, connectivity_edges[:, 0], edge_lengths)
    np.minimum.at(vertex_minimum, connectivity_edges[:, 1], edge_lengths)
    np.maximum.at(vertex_maximum, connectivity_edges[:, 0], edge_lengths)
    np.maximum.at(vertex_maximum, connectivity_edges[:, 1], edge_lengths)
    active = np.isfinite(vertex_minimum) & (vertex_minimum > 0.0)
    maximum_local_edge_ratio = float(
        np.max(vertex_maximum[active] / vertex_minimum[active], initial=1.0)
    )
    compliance_issues = []
    if (
        not set(family_policy.required) <= achieved_kinds
        or not achieved_kinds <= requested_kinds
        or (len(achieved_kinds) > 1 and not family_policy.allow_mixed)
    ):
        compliance_issues.append("cell_family")
    size_requested = [
        (
            "size_compliance_absolute_tolerance",
            specification.size_compliance.absolute_tolerance,
        ),
        (
            "size_compliance_relative_tolerance",
            specification.size_compliance.relative_tolerance,
        ),
    ]
    size_achieved = []
    if semantic_surface:
        size_requested.extend(
            (
                ("region_count", float(len(specification.region_controls))),
                (
                    "required_patch_count",
                    float(
                        sum(control.required for control in specification.patch_controls)
                    ),
                ),
            )
        )
        size_achieved.extend(
            (
                ("region_count", float(len(zones))),
                ("patch_count", float(len(patches))),
            )
        )
    if semantic_volume:
        size_issues, local_requested, local_achieved = _semantic_size_compliance(
            mesh, specification, cell_solid_ids
        )
        compliance_issues.extend(size_issues)
        size_requested.extend(local_requested)
        size_requested.extend(
            (
                ("region_count", float(len(specification.region_controls))),
                (
                    "required_patch_count",
                    float(
                        sum(control.required for control in specification.patch_controls)
                    ),
                ),
            )
        )
        size_achieved.extend(local_achieved)
        size_achieved.extend(
            (
                ("region_count", float(len(region_zones))),
                ("patch_count", float(len(patches))),
                ("size_field_count", float(len(size_field_ids))),
            )
        )
    else:
        top_scope = (
            specification.scope
            if isinstance(specification, SurfaceMeshingSpec)
            else specification.boundary_scope
        )
        for control in specification.size_controls:
            if (
                isinstance(control, ProximitySizeControl)
                or control.scope.scope_id != top_scope.scope_id
            ):
                continue
            if isinstance(control, UniformSizeControl):
                size_issues, local_requested, local_achieved = _edge_size_evidence(
                    control,
                    connectivity_edges,
                    np.asarray(mesh.coordinates, dtype=np.float64),
                    specification,
                )
                compliance_issues.extend(size_issues)
                size_requested.extend(local_requested)
                size_achieved.extend(local_achieved)
                continue
            key = f"size:{control.control_id}"
            size_requested.append((f"{key}:normal_angle", control.normal_angle))
            optional_bounds = (
                ("minimum_size", control.minimum_size),
                ("maximum_size", control.maximum_size),
            )
            size_requested.extend(
                (f"{key}:{name}", value)
                for name, value in optional_bounds
                if value is not None
            )
            size_achieved.extend(
                (
                    (f"{key}:minimum_edge", minimum_edge),
                    (f"{key}:maximum_edge", maximum_edge),
                )
            )
            if control.strength is SizeControlStrength.HARD:
                policy = specification.size_compliance
                if control.minimum_size is not None:
                    tolerance = policy.absolute_tolerance + (
                        policy.relative_tolerance * abs(control.minimum_size)
                    )
                    if minimum_edge < control.minimum_size - tolerance:
                        compliance_issues.append(f"minimum_size:{control.control_id}")
                if control.maximum_size is not None:
                    tolerance = policy.absolute_tolerance + (
                        policy.relative_tolerance * abs(control.maximum_size)
                    )
                    if maximum_edge > control.maximum_size + tolerance:
                        compliance_issues.append(f"maximum_size:{control.control_id}")
        size_achieved.append(("size_field_count", float(len(size_field_ids))))
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(compliance_issues),
        requested=(
            *size_requested,
            *periodic_requested,
            *layer_audit.requested,
            *band_requested,
        ),
        achieved=(
            ("minimum_edge", minimum_edge),
            ("maximum_edge", maximum_edge),
            ("minimum_curved_jacobian_determinant", minimum_jacobian),
            ("maximum_local_edge_ratio", maximum_local_edge_ratio),
            *size_achieved,
            *periodic_achieved,
            *layer_audit.achieved,
            *band_achieved,
            *layer_interface_achieved,
        ),
    )
    return audit, compliance


class GmshProvider:
    def __init__(self, options: GmshOptions | None = None, /):
        self.options = GmshOptions() if options is None else options
        if not isinstance(self.options, GmshOptions):
            raise TypeError("options must be GmshOptions or None.")

    @property
    def info(self) -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "gmsh",
            "runtime",
            "GPL-2.0-or-later",
            operations=(MeshingOperation.MESH_SURFACE, MeshingOperation.MESH_VOLUME),
            source_kinds=(MeshingSourceKind.BREP,),
            capabilities=(
                MeshingCapability.DETERMINISTIC,
                MeshingCapability.CAD_CONFORMING,
                MeshingCapability.HIGH_ORDER_GEOMETRY,
                MeshingCapability.PERIODIC,
                MeshingCapability.MIXED_CELLS,
                MeshingCapability.BOUNDARY_LAYERS,
                MeshingCapability.MULTI_MATERIAL,
            ),
            cell_kinds=(
                "triangle",
                "quadrilateral",
                "tetrahedron",
                "prism",
                "hexahedron",
            ),
            dimensions=(2, 3),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    def entity_scope(
        self,
        source: _BRepMeshingSource,
        entity_ids: Sequence[BRepEntityId] | BRepEntityId,
        /,
    ) -> MeshingScope:
        model = _brep_model(source)
        entities = (
            (entity_ids,) if isinstance(entity_ids, BRepEntityId) else tuple(entity_ids)
        )
        if not entities or not all(
            isinstance(entity, BRepEntityId) for entity in entities
        ):
            raise TypeError("entity_ids must contain at least one BRepEntityId.")
        revisions = {entity.source_revision for entity in entities}
        kinds = {entity.kind for entity in entities}
        if revisions != {model.report.source_revision} or len(kinds) != 1:
            raise ValueError(
                "BRep entity scopes require one kind from the supplied source revision."
            )
        kind = kinds.pop()
        dimensions = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
        counts = {
            "vertex": model.report.num_vertices,
            "edge": model.report.num_edges,
            "face": model.report.num_faces,
            "solid": model.topology.num_solids,
        }
        if kind not in dimensions:
            raise ValueError(
                "Gmsh BRep scopes support vertices, edges, faces, and solids."
            )
        identifiers = np.asarray(
            tuple(entity.index for entity in entities), dtype=np.int64
        )
        if (
            np.any(identifiers < 0)
            or np.any(identifiers >= counts[kind])
            or np.unique(identifiers).size != identifiers.size
        ):
            raise ValueError("BRep entity scope contains an out-of-range or repeated ID.")
        dimension = dimensions[kind]
        return MeshingScope(
            model.report.source_id,
            model.report.source_revision,
            MeshingEntityKind.GEOMETRY,
            dimension,
            f"{model.report.source_revision}:brep:{dimension}",
            identifiers,
        )

    def whole_scope(self, source: _BRepMeshingSource, dimension: int, /) -> MeshingScope:
        model = _brep_model(source)
        target = int(dimension)
        entities = {
            1: model.edge_ids,
            2: model.face_ids,
            3: model.solid_ids,
        }
        if target not in entities:
            raise ValueError("Gmsh BRep scope dimension must be one, two, or three.")
        if not entities[target]:
            raise ValueError(
                f"The BRep source contains no dimension-{target} entities to scope."
            )
        return self.entity_scope(model, entities[target])

    def inspect_source(self, source: _BRepMeshingSource, /) -> MeshingSourceDescriptor:
        model = _brep_model(source)
        closed = model.topology.num_solids > 0
        return MeshingSourceDescriptor(
            model.report.source_id,
            model.report.source_revision,
            MeshingSourceKind.BREP,
            3 if closed else 2,
            3,
            closed=closed,
        )

    def validate(
        self,
        source: _BRepMeshingSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
    ) -> ProviderSupportReport:
        model = _brep_model(source)
        descriptor = self.inspect_source(source)
        unsupported = []
        target = specification.target
        scope = (
            specification.scope
            if isinstance(specification, SurfaceMeshingSpec)
            else specification.boundary_scope
        )
        dimension = target.topological_dimension
        scope_ids = _scope_indices(model, scope, dimension)
        expected_ids = np.arange(
            model.topology.num_solids if dimension == 3 else model.report.num_faces,
            dtype=np.int64,
        )
        if scope_ids is None:
            unsupported.append(
                "top-level scope does not bind the exact supplied BRep entity set"
            )
        elif not np.array_equal(scope_ids, expected_ids):
            unsupported.append(
                "Gmsh meshes the complete source; partial top-level scopes are unsupported"
            )
        if isinstance(specification, SurfaceMeshingSpec):
            if target.ambient_dimension == 2 and model.topology.num_solids:
                unsupported.append(
                    "Ambient-dimension-two output requires a zero-solid planar BRep"
                )
            if target.ambient_dimension not in (2, 3):
                unsupported.append(
                    "Gmsh surface output supports ambient dimensions two and three"
                )
        elif target.ambient_dimension != 3:
            unsupported.append(
                "Gmsh volume output requires three-dimensional source coordinates"
            )
        requested = {
            *target.cell_families.required,
            *target.cell_families.preferred,
            *target.cell_families.allowed_transitions,
        }
        layers = (
            specification.layer_controls
            if isinstance(specification, VolumeMeshingSpec)
            else ()
        )
        supported = (
            {"triangle", "quadrilateral"}
            if dimension == 2
            else {"prism", "tetrahedron"}
            if layers
            else {"tetrahedron"}
        )
        if not requested <= supported:
            unsupported.append(f"Gmsh selected path supports only {sorted(supported)}")
        if target.geometry_order not in (1, 2):
            unsupported.append(
                "Gmsh canonical complete geometry elements support orders one and two"
            )
        if specification.protected_features:
            unsupported.append(
                "Gmsh protected-feature deviation contracts are not implemented"
            )

        semantic_volume = _validate_gmsh_semantics(
            source,
            specification,
            model,
            descriptor,
            target,
            scope,
            requested,
            layers,
            unsupported,
        )
        _validate_gmsh_periodicity(specification, model, unsupported)
        _validate_gmsh_size_controls(
            specification, model, scope, semantic_volume, unsupported
        )
        return ProviderSupportReport(
            self.info,
            descriptor,
            specification,
            unsupported=tuple(dict.fromkeys(unsupported)),
        )

    def plan(
        self,
        source: _BRepMeshingSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
    ) -> GmshMeshingPlan:
        return GmshMeshingPlan(
            source,
            specification,
            self.options,
            self.validate(source, specification),
        )

    def open_session(
        self,
        policy: MeshingExecutionPolicy | None = None,
        /,
    ) -> GmshSession:
        execution = MeshingExecutionPolicy() if policy is None else policy
        if not isinstance(execution, MeshingExecutionPolicy):
            raise TypeError("policy must be MeshingExecutionPolicy or None.")
        return GmshSession(self, execution)

    def execute(self, plan: GmshMeshingPlan, /) -> CellMeshingResult:
        started = time.perf_counter()
        emit(
            "DEBUG",
            "provider.execution.started",
            "Gmsh execution started",
            plan_id=plan.plan_id,
            provider="gmsh",
        )
        try:
            with self.open_session() as session:
                result = session.execute(plan)
        except MeshingFailure as error:
            emit(
                "ERROR",
                "provider.execution.failed",
                "Gmsh execution failed",
                elapsed_seconds=time.perf_counter() - started,
                failure_category=error.category.value,
                plan_id=plan.plan_id,
                provider="gmsh",
            )
            raise
        emit(
            "INFO",
            "provider.execution.completed",
            "Gmsh execution completed",
            elapsed_seconds=time.perf_counter() - started,
            mesh_id=result.mesh.mesh_id,
            plan_id=plan.plan_id,
            provider="gmsh",
            trace_id=result.trace.trace_id,
        )
        return result


@dataclass(frozen=True, slots=True)
class _ElementRows:
    tags: np.ndarray
    vertices: np.ndarray
    entity_tags: np.ndarray
    element_type: int
    cell_kind: str
    corner_count: int

    @property
    def block_name(self) -> str:
        return {
            "triangle": "triangles",
            "quadrilateral": "quadrilaterals",
            "tetrahedron": "tetrahedra",
            "prism": "prisms",
            "hexahedron": "hexahedra",
        }[self.cell_kind]


def _element_rows(
    gmsh, dimension: int, geometry_order: int, /
) -> tuple[_ElementRows, ...]:
    kinds = {
        "Triangle": "triangle",
        "Quadrilateral": "quadrilateral",
        "Tetrahedron": "tetrahedron",
        "Prism": "prism",
        "Hexahedron": "hexahedron",
    }
    records: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    properties = {}
    for _, entity_tag in sorted(gmsh.model.getEntities(dimension)):
        element_types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(
            dimension, entity_tag
        )
        for element_type, tag_values, node_values in zip(
            element_types, tag_blocks, node_blocks, strict=True
        ):
            element_type = int(element_type)
            name, _, order, count, _, corners = gmsh.model.mesh.getElementProperties(
                element_type
            )
            family = name.split()[0]
            if family not in kinds or int(order) != geometry_order:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    f"Gmsh returned unsupported element {name!r}; no elements may be discarded.",
                    stage=MeshingStageKind.CANONICALIZATION.value,
                )
            tags = np.asarray(tag_values, dtype=np.int64)
            nodes = np.asarray(node_values, dtype=np.int64).reshape((-1, int(count)))
            records.setdefault(element_type, []).append(
                (tags, nodes, np.full(tags.shape, entity_tag, dtype=np.int64))
            )
            properties[element_type] = (kinds[family], int(corners))
    if not records:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            f"Gmsh returned no dimension-{dimension} elements.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    result = []
    for element_type, chunks in records.items():
        tags, nodes, entities = (
            np.concatenate(values) for values in zip(*chunks, strict=True)
        )
        order = np.argsort(tags, kind="stable")
        kind, corners = properties[element_type]
        result.append(
            _ElementRows(
                tags[order], nodes[order], entities[order], element_type, kind, corners
            )
        )
    key = (
        (lambda rows: (rows.corner_count, rows.block_name))
        if dimension == 2
        else (lambda rows: rows.block_name)
    )
    return tuple(sorted(result, key=key))


def _local_connectivity(node_tags: np.ndarray, values: np.ndarray, /) -> np.ndarray:
    locations = np.searchsorted(node_tags, values)
    if np.any(locations >= node_tags.size) or not np.array_equal(
        node_tags[locations], values
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh element connectivity references an undeclared node tag.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return locations.astype(np.int32, copy=False)


def _geometry_permutation(gmsh, rows: _ElementRows, element, /) -> np.ndarray:
    """Map actual Gmsh reference nodes, not meshio's distinct wedge/hex ordering."""
    _, dimension, _, count, coordinates, _ = gmsh.model.mesh.getElementProperties(
        rows.element_type
    )
    source = np.asarray(coordinates, dtype=np.float64).reshape(
        (int(count), int(dimension))
    )
    if rows.cell_kind in ("quadrilateral", "hexahedron"):
        source = 0.5 * (source + 1.0)
    elif rows.cell_kind == "prism":
        source[:, 2] = 0.5 * (source[:, 2] + 1.0)
    target = np.asarray(element.reference_nodes, dtype=np.float64)
    matches = np.max(np.abs(target[:, None] - source[None]), axis=-1) <= 2.0e-12
    if source.shape != target.shape or not np.all(np.sum(matches, axis=1) == 1):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            f"Gmsh {rows.cell_kind} geometry nodes do not match the canonical complete element.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return np.argmax(matches, axis=1).astype(np.int32)


def _audit_jacobians(gmsh, rows: tuple[_ElementRows, ...], /) -> float:
    """Audit the curved map with Gmsh's adaptive determinant extrema, not corners."""
    minimum = np.inf
    for block in rows:
        determinants = np.asarray(
            gmsh.model.mesh.getElementQualities(block.tags, "minDetJac"), dtype=np.float64
        )
        invalid = ~np.isfinite(determinants) | (determinants <= 0.0)
        if determinants.shape != block.tags.shape or np.any(invalid):
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "Gmsh curved-element minimum Jacobian determinant is nonpositive or unavailable.",
                stage=MeshingStageKind.GEOMETRY_AUDIT.value,
                entity_ids=tuple(block.tags[invalid])
                if determinants.shape == block.tags.shape
                else (),
            )
        minimum = min(minimum, float(np.min(determinants)))
    return minimum


@runtime_checkable
class _TopoDSEdgeCaster(Protocol):
    @staticmethod
    def Edge_s(shape: TopoDS_Shape, /) -> TopoDS_Edge: ...


def _scope_samples(
    source: BRepModel, shape, scope: MeshingScope, /
) -> tuple[np.ndarray, ...]:
    """Sample stable source entities independently of Gmsh's import tag numbering."""
    ids = np.asarray(scope.entity_ids, dtype=np.int64)
    counts = {
        0: source.report.num_vertices,
        1: source.report.num_edges,
        2: source.report.num_faces,
    }
    if (
        scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension not in counts
        or scope.source_id != source.report.source_id
        or scope.source_revision != source.report.source_revision
        or np.any(ids >= counts[scope.entity_dimension])
    ):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Scope does not select an entity of the supplied BRep revision.",
        )
    if scope.entity_dimension == 2:
        face_ids = np.asarray(source.triangle_face_ids)
        parameters = np.asarray(source.triangle_parameters)
        result = []
        for face in ids:
            triangles = np.flatnonzero(face_ids == face)
            if not triangles.size:
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Source face has no interior samples for entity resolution.",
                )
            selected = triangles[
                np.linspace(0, len(triangles) - 1, min(3, len(triangles)), dtype=np.int64)
            ]
            uv = np.mean(parameters[selected], axis=1)
            result.append(np.asarray(source.patches[int(face)].evaluate(jnp.asarray(uv))))
        return tuple(result)
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    from ...geometry.brep._occt import _explore_unique

    if scope.entity_dimension == 0:
        vertices = _explore_unique(shape, TopAbs_VERTEX, TopoDS.Vertex_s)
        result = []
        for vertex in ids:
            point = BRep_Tool.Pnt_s(vertices[int(vertex)])
            result.append(np.asarray(((point.X(), point.Y(), point.Z()),)))
        return tuple(result)
    if not isinstance(TopoDS, _TopoDSEdgeCaster):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "The CAD kernel must expose the TopoDS.Edge_s edge downcast.",
        )
    edges = _explore_unique(shape, TopAbs_EDGE, TopoDS.Edge_s)
    result = []
    for edge in ids:
        curve = BRepAdaptor_Curve(edges[int(edge)])
        parameters = np.linspace(curve.FirstParameter(), curve.LastParameter(), 5)[1:-1]
        values = [curve.Value(float(value)) for value in parameters]
        result.append(np.asarray([(point.X(), point.Y(), point.Z()) for point in values]))
    return tuple(result)


def _match_entities(
    gmsh, dimension: int, samples, candidates, tolerance: float, /
) -> tuple[int, ...]:
    result = []
    for points in samples:
        matches = []
        for tag in candidates:
            if tag in result:
                continue
            closest, _ = gmsh.model.getClosestPoint(
                dimension, tag, np.asarray(points).reshape(-1)
            )
            closest = np.asarray(closest).reshape((-1, 3))
            if (
                closest.shape == points.shape
                and np.max(np.linalg.norm(closest - points, axis=1)) <= tolerance
                and gmsh.model.isInside(dimension, tag, closest.reshape(-1))
                == len(points)
            ):
                matches.append(tag)
        if len(matches) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "BRep entity does not resolve uniquely to the imported Gmsh geometry.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        result.append(matches[0])
    return tuple(result)


def _resolve_entities(gmsh, source, shape, scope, /) -> tuple[int, ...]:
    samples = _scope_samples(source, shape, scope)
    scale = max(float(np.ptp(np.asarray(source.mesh_vertices), axis=0).max()), 1.0)
    return _match_entities(
        gmsh,
        scope.entity_dimension,
        samples,
        [tag for _, tag in gmsh.model.getEntities(scope.entity_dimension)],
        1.0e-7 * scale,
    )


@dataclass(frozen=True, slots=True)
class _CadEntityMap:
    face_to_surface: tuple[int, ...]
    solid_to_volume: tuple[int, ...]
    edge_to_curve: tuple[int, ...] = ()


def _validate_source_solids(source: BRepModel, shape, /) -> tuple:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS

    topology = source.topology
    if (
        any(not faces for faces in topology.solid_faces)
        or any(len(owners) not in (1, 2) for owners in topology.face_solids)
        or any(
            orientations[0] == orientations[1]
            for face_index, owners in enumerate(topology.face_solids)
            if len(owners) == 2
            for orientations in (
                tuple(
                    topology.solid_face_orientations[solid][
                        topology.solid_faces[solid].index(face_index)
                    ]
                    for solid in owners
                ),
            )
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Source solids require nonempty manifold boundaries with opposite shared-face orientations.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    solids = _explore_unique(shape, TopAbs_SOLID, TopoDS.Solid_s)
    if len(solids) != topology.num_solids or any(
        not BRepCheck_Analyzer(solid).IsValid() for solid in solids
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Reopened BRep solids do not match the valid imported solid inventory.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    scale = max(float(np.ptp(np.asarray(source.mesh_vertices), axis=0).max()), 1.0)
    volume_tolerance = 1.0e-12 * scale**3
    contact_tolerance = 1.0e-9 * scale
    for left_index, left in enumerate(solids):
        left_faces = set(topology.solid_faces[left_index])
        for right_index, right in enumerate(
            solids[left_index + 1 :], start=left_index + 1
        ):
            distance = BRepExtrema_DistShapeShape(left, right)
            distance.Perform()
            if not distance.IsDone():
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid contact validation did not complete.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            shared_faces = left_faces & set(topology.solid_faces[right_index])
            if not shared_faces and float(distance.Value()) <= contact_tolerance:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solids touch without a complete shared topological face.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            common = BRepAlgoAPI_Common(left, right)
            common.Build()
            if not common.IsDone():
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid overlap validation did not complete.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            properties = GProp_GProps()
            BRepGProp.VolumeProperties_s(common.Shape(), properties)
            if abs(float(properties.Mass())) > volume_tolerance:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid interiors overlap; provider-side ownership fragmentation is prohibited.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
    return solids


def _resolve_planar_cad_entity_map(gmsh, source: BRepModel, shape, /) -> _CadEntityMap:
    if source.topology.num_solids:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Strict planar CAD meshing requires a zero-solid BRep.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    face_scope = MeshingScope(
        source.report.source_id,
        source.report.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        f"{source.report.source_revision}:brep:2",
        np.arange(source.report.num_faces, dtype=np.int64),
    )
    edge_scope = MeshingScope(
        source.report.source_id,
        source.report.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        f"{source.report.source_revision}:brep:1",
        np.arange(source.report.num_edges, dtype=np.int64),
    )
    face_tags = _resolve_entities(gmsh, source, shape, face_scope)
    edge_tags = _resolve_entities(gmsh, source, shape, edge_scope)
    if set(face_tags) != {tag for _, tag in gmsh.model.getEntities(2)}:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh surfaces are not a bijection with planar BRep faces.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    if set(edge_tags) != {tag for _, tag in gmsh.model.getEntities(1)}:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh curves are not a bijection with planar BRep edges.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    for face_index, surface in enumerate(face_tags):
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
        actual = {abs(int(tag)) for dimension, tag in boundary if dimension == 1}
        expected = {edge_tags[index] for index in source.topology.face_edges[face_index]}
        if len(actual) != len(boundary) or actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh planar surface boundaries differ from source face-edge incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
    for edge_index, curve in enumerate(edge_tags):
        upward, _ = gmsh.model.getAdjacencies(1, curve)
        actual = {int(value) for value in np.asarray(upward, dtype=np.int64)}
        expected = {face_tags[index] for index in source.topology.edge_faces[edge_index]}
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh curve-to-surface adjacency differs from source planar incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
    return _CadEntityMap(tuple(face_tags), (), tuple(edge_tags))


def _resolve_cad_entity_map(gmsh, source: BRepModel, shape, /) -> _CadEntityMap:
    _validate_source_solids(source, shape)
    face_scope = MeshingScope(
        source.report.source_id,
        source.report.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        f"{source.report.source_revision}:brep:2",
        np.arange(source.report.num_faces, dtype=np.int64),
    )
    face_tags = _resolve_entities(gmsh, source, shape, face_scope)
    imported_surfaces = {tag for _, tag in gmsh.model.getEntities(2)}
    if set(face_tags) != imported_surfaces:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh surfaces are not a bijection with source BRep faces.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )

    volume_tags = tuple(tag for _, tag in sorted(gmsh.model.getEntities(3)))
    if len(volume_tags) != source.topology.num_solids:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh volumes do not match the source solid inventory.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    boundary_faces: dict[int, frozenset[int]] = {}
    boundary_orientations: dict[tuple[int, int], int] = {}
    for volume in volume_tags:
        occurrences = gmsh.model.getBoundary(
            [(3, volume)], combined=False, oriented=True, recursive=False
        )
        if any(dimension != 2 for dimension, _ in occurrences):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An imported Gmsh volume has a non-surface boundary occurrence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        tags = tuple(abs(int(tag)) for _, tag in occurrences)
        if len(set(tags)) != len(tags):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An imported Gmsh volume repeats a boundary surface.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        boundary_faces[volume] = frozenset(tags)
        boundary_orientations.update(
            (
                ((volume, abs(int(tag))), 1 if int(tag) > 0 else -1)
                for _, tag in occurrences
            )
        )

    solid_to_volume = []
    used_volumes = set()
    for faces in source.topology.solid_faces:
        expected = frozenset(face_tags[face] for face in faces)
        candidates = tuple(
            volume
            for volume, actual in boundary_faces.items()
            if actual == expected and volume not in used_volumes
        )
        if len(candidates) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "A source solid does not resolve uniquely by exact boundary-face incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        solid_to_volume.append(candidates[0])
        used_volumes.add(candidates[0])
    if used_volumes != set(volume_tags):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh volume ownership is not a source-solid bijection.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )

    for face_index, surface in enumerate(face_tags):
        source_owners = source.topology.face_solids[face_index]
        expected_volumes = {solid_to_volume[owner] for owner in source_owners}
        upward, _ = gmsh.model.getAdjacencies(2, surface)
        actual_volumes = {int(value) for value in np.asarray(upward, dtype=np.int64)}
        if actual_volumes != expected_volumes:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh surface-to-volume adjacency differs from source BRep incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        if len(source_owners) == 2:
            first, second = tuple(expected_volumes)
            if (
                boundary_orientations[first, surface]
                == boundary_orientations[second, surface]
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "A shared Gmsh surface has inconsistent volume orientations.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
    return _CadEntityMap(tuple(face_tags), tuple(solid_to_volume))


@dataclass(frozen=True, slots=True)
class _PlanarBandFront:
    layer: object
    curves: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PlanarBandGeneration:
    embedding: PlanarEmbedding
    fronts: tuple[_PlanarBandFront, ...]


def _straight_surface_curves(
    gmsh,
    surface: int,
    embedding: PlanarEmbedding,
    /,
) -> tuple[tuple[int, ...], dict[int, np.ndarray]]:
    boundary = gmsh.model.getBoundary(
        [(2, surface)], combined=False, oriented=False, recursive=False
    )
    curves = tuple(abs(int(tag)) for dimension, tag in boundary if dimension == 1)
    if len(curves) != 4 or len(curves) != len(boundary):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Pure quadrilateral planar bands require every closure face to have four curves.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    directions: dict[int, np.ndarray] = {}
    for curve in curves:
        if str(gmsh.model.getType(1, curve)).lower() != "line":
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Pure quadrilateral planar bands require straight closure curves.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        vertices = gmsh.model.getBoundary(
            [(1, curve)], combined=False, oriented=False, recursive=False
        )
        if len(vertices) != 2 or any(dimension != 0 for dimension, _ in vertices):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band boundary curve is not one straight segment.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        coordinates = np.asarray(
            [gmsh.model.getValue(0, abs(int(tag)), []) for _, tag in vertices],
            dtype=np.float64,
        )
        planar = embedding.to_planar(coordinates)
        direction = planar[1] - planar[0]
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band closure curve has zero length.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        directions[curve] = direction / length
    return curves, directions


def _configure_full_quad_band_closure(
    gmsh,
    embedding: PlanarEmbedding,
    constrained_curves: dict[int, int],
    target_size: float | None,
    /,
) -> None:
    """Propagate exact band node counts across rectangular Q4 closure faces."""

    surfaces = tuple(
        int(tag) for dimension, tag in gmsh.model.getEntities(2) if dimension == 2
    )
    boundaries = {
        surface: _straight_surface_curves(gmsh, surface, embedding)
        for surface in surfaces
    }
    parallel_pairs: dict[int, tuple[tuple[int, int], ...]] = {}
    for surface, (curves, directions) in boundaries.items():
        remaining = set(curves)
        pairs: list[tuple[int, int]] = []
        while remaining:
            first = min(remaining)
            matches = tuple(
                second
                for second in remaining - {first}
                if abs(float(np.dot(directions[first], directions[second])))
                >= 1.0 - 1.0e-10
            )
            if len(matches) != 1:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Pure quadrilateral planar bands require rectangular closure faces.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            second = matches[0]
            remaining.remove(first)
            remaining.remove(second)
            pairs.append((first, second))
        parallel_pairs[surface] = tuple(pairs)
    neighbors: dict[int, set[int]] = {
        curve: set() for curves, _ in boundaries.values() for curve in curves
    }
    for pairs in parallel_pairs.values():
        for first, second in pairs:
            neighbors[first].add(second)
            neighbors[second].add(first)
    unresolved = set(neighbors)
    while unresolved:
        root = min(unresolved)
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for neighbor in neighbors[current] - component:
                component.add(neighbor)
                frontier.append(neighbor)
        unresolved -= component
        prescribed = {
            constrained_curves[curve]
            for curve in component
            if curve in constrained_curves
        }
        if len(prescribed) > 1:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Pure quadrilateral planar band closure has incompatible opposite curve counts.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        if prescribed:
            count = next(iter(prescribed))
        elif target_size is None:
            count = 2
        else:
            lengths = []
            for curve in component:
                vertices = gmsh.model.getBoundary(
                    [(1, curve)], combined=False, oriented=False, recursive=False
                )
                coordinates = np.asarray(
                    [gmsh.model.getValue(0, abs(int(tag)), []) for _, tag in vertices],
                    dtype=np.float64,
                )
                planar = embedding.to_planar(coordinates)
                lengths.append(float(np.linalg.norm(planar[1] - planar[0])))
            count = max(2, int(np.ceil(max(lengths) / target_size)) + 1)
        for curve in component:
            constrained_curves[curve] = count
    for surface, (curves, _) in boundaries.items():
        for curve in curves:
            gmsh.model.mesh.setTransfiniteCurve(curve, constrained_curves[curve])
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.setRecombine(2, surface)


def _apply_planar_band_constraints(
    gmsh,
    bands: PlanarBandResult | None,
    cad_entities: _CadEntityMap | None,
    requested_kinds: set[str],
    target_size: float | None,
    /,
) -> _PlanarBandGeneration | None:
    if bands is None:
        return None
    if cad_entities is None or not cad_entities.edge_to_curve:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Planar bands require resolved source face and edge identities.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    constrained_curves: dict[int, int] = {}
    fronts = []
    for layer in bands.layer_partitions:
        if len(layer.face_entity_ids) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Each exact planar band layer must be one strip face.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        surface = cad_entities.face_to_surface[layer.face_entity_ids[0].index]
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
        curves = tuple(abs(int(tag)) for dimension, tag in boundary if dimension == 1)
        if len(curves) != 4 or len(curves) != len(boundary):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band layer is not an exact four-curve straight strip.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        tangential_nodes = max(
            2, int(np.ceil(layer.source_length / layer.tangential_target)) + 1
        )
        tangent = np.asarray(layer.tangent, dtype=np.float64)
        tangential_count = 0
        normal_count = 0
        for curve in curves:
            points = gmsh.model.getBoundary(
                [(1, curve)], combined=False, oriented=False, recursive=False
            )
            if len(points) != 2 or any(dimension != 0 for dimension, _ in points):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "A planar band boundary curve is not one straight segment.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            coordinates = np.asarray(
                [gmsh.model.getValue(0, abs(int(tag)), []) for _, tag in points],
                dtype=np.float64,
            )
            planar = bands.embedding.to_planar(coordinates)
            direction = planar[1] - planar[0]
            direction /= np.linalg.norm(direction)
            is_tangential = abs(float(np.dot(direction, tangent))) >= 1.0 - 1.0e-10
            node_count = tangential_nodes if is_tangential else 2
            previous = constrained_curves.setdefault(curve, node_count)
            if previous != node_count:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Intersecting planar band constraints require incompatible curve nodes.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            gmsh.model.mesh.setTransfiniteCurve(curve, node_count)
            tangential_count += int(is_tangential)
            normal_count += int(not is_tangential)
        if tangential_count != 2 or normal_count != 2:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band strip lacks two tangential and two normal curves.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        gmsh.model.mesh.setTransfiniteSurface(surface)
        if "quadrilateral" in requested_kinds:
            gmsh.model.mesh.setRecombine(2, surface)
        front_patch = bands.partition.patch(layer.front_patch_name)
        front_curves = tuple(
            cad_entities.edge_to_curve[entity.index] for entity in front_patch.entity_ids
        )
        if not front_curves:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An exact planar band front resolved to no source curves.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        fronts.append(_PlanarBandFront(layer, front_curves))
    if requested_kinds == {"quadrilateral"}:
        _configure_full_quad_band_closure(
            gmsh,
            bands.embedding,
            constrained_curves,
            target_size,
        )
    return _PlanarBandGeneration(bands.embedding, tuple(fronts))


def _audit_planar_band_fronts(
    gmsh, generation: _PlanarBandGeneration | None, /
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    if generation is None:
        return (), ()
    requested = []
    achieved = []
    for record in generation.fronts:
        layer = record.layer
        coordinates = []
        for curve in record.curves:
            _, values, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
            coordinates.append(np.asarray(values, dtype=np.float64).reshape((-1, 3)))
        points = np.concatenate(coordinates)
        if not points.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "An exact planar band front contains no mesh nodes.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        planar = generation.embedding.to_planar(points)
        origin = np.asarray(layer.source_origin, dtype=np.float64)
        inward = np.asarray(layer.inward_normal, dtype=np.float64)
        distances = (planar - origin) @ inward
        residual = float(
            np.max(np.abs(distances - layer.cumulative_distance), initial=0.0)
        )
        tangent = np.asarray(layer.tangent, dtype=np.float64)
        tangential_positions = np.unique((planar - origin) @ tangent)
        if tangential_positions.size < 2:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "An exact planar band front has fewer than two tangential nodes.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        maximum_tangential_spacing = float(np.max(np.diff(tangential_positions)))
        scale = max(
            1.0,
            layer.source_length,
            layer.cumulative_distance,
            float(np.max(np.abs(planar), initial=0.0)),
        )
        tolerance = 8192.0 * np.finfo(np.float64).eps * scale
        if residual > tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Generated planar band nodes do not lie on the exact requested front.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        if maximum_tangential_spacing > layer.tangential_target + tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Generated planar band tangential spacing exceeds its target.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        key = f"planar_band:{layer.control_id}:{layer.region_name}:front:{layer.layer_index + 1}"
        requested.extend(
            (
                (f"{key}:distance", layer.cumulative_distance),
                (f"{key}:tangential_target", layer.tangential_target),
            )
        )
        achieved.extend(
            (
                (f"{key}:distance", float(np.mean(distances))),
                (f"{key}:maximum_residual", residual),
                (
                    f"{key}:maximum_tangential_spacing",
                    maximum_tangential_spacing,
                ),
            )
        )
    return tuple(requested), tuple(achieved)


def _set_periodic(gmsh, plan, shape, /):
    records = []
    slaves_used = set()
    for constraint in plan.specification.periodic_constraints:
        dimension = constraint.source_scope.entity_dimension
        masters = _resolve_entities(gmsh, plan.source, shape, constraint.source_scope)
        candidates = _resolve_entities(gmsh, plan.source, shape, constraint.target_scope)
        transform = np.asarray(constraint.transform)
        samples = _scope_samples(plan.source, shape, constraint.source_scope)
        transformed = tuple(
            points @ transform[:3, :3].T + transform[:3, 3] for points in samples
        )
        slaves = _match_entities(
            gmsh, dimension, transformed, candidates, constraint.tolerance
        )
        if set(masters) & set(slaves) or any(
            (dimension, tag) in slaves_used for tag in slaves
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Periodic self-pairs and multiply constrained slave entities are unsupported.",
            )
        gmsh.model.mesh.setPeriodic(
            dimension, list(slaves), list(masters), transform.reshape(-1).tolist()
        )
        slaves_used.update((dimension, tag) for tag in slaves)
        records.append((constraint, masters, slaves))
    return tuple(records)


def _audit_periodic(gmsh, records, node_tags, points, /):
    requested = []
    achieved = []
    for constraint, masters, slaves in records:
        dimension = constraint.source_scope.entity_dimension
        transform = np.asarray(constraint.transform)
        residual = 0.0
        pair_count = 0
        for master, slave in zip(masters, slaves, strict=True):
            actual_master, slave_nodes, master_nodes, actual_transform = (
                gmsh.model.mesh.getPeriodicNodes(
                    dimension, slave, includeHighOrderNodes=True
                )
            )
            slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
            master_nodes = np.asarray(master_nodes, dtype=np.int64)
            expected_slave, _, _ = gmsh.model.mesh.getNodes(
                dimension, slave, includeBoundary=True
            )
            expected_master, _, _ = gmsh.model.mesh.getNodes(
                dimension, master, includeBoundary=True
            )
            if (
                actual_master != master
                or not np.array_equal(np.sort(slave_nodes), np.unique(expected_slave))
                or not np.array_equal(np.sort(master_nodes), np.unique(expected_master))
                or not np.allclose(
                    np.asarray(actual_transform).reshape((4, 4)),
                    transform,
                    rtol=0.0,
                    atol=constraint.tolerance,
                )
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Gmsh periodic correspondence is not a complete high-order node bijection.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            slave_points = points[_local_connectivity(node_tags, slave_nodes)]
            master_points = points[_local_connectivity(node_tags, master_nodes)]
            mapped = master_points @ transform[:3, :3].T + transform[:3, 3]
            residual = max(
                residual,
                float(np.max(np.linalg.norm(slave_points - mapped, axis=1), initial=0.0)),
            )
            pair_count += slave_nodes.size
        if residual > constraint.tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Periodic node residual exceeds the exact requested tolerance.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        key = f"periodic:{constraint.constraint_id}"
        requested.append((f"{key}:tolerance", constraint.tolerance))
        achieved.extend(
            (
                (f"{key}:maximum_residual", residual),
                (f"{key}:node_pairs", float(pair_count)),
            )
        )
    return tuple(requested), tuple(achieved)


@dataclass(frozen=True, slots=True)
class _SweptVolume:
    control_index: int
    control: SweptLayerControl
    solid_index: int
    volume_tag: int
    source_face_index: int
    target_face_index: int
    source_surface: int
    target_surface: int
    lateral_face_indices: tuple[int, ...]
    lateral_surfaces: tuple[int, ...]
    origin: np.ndarray
    direction: np.ndarray
    unit: np.ndarray
    levels: np.ndarray
    relative_cad_difference: float


@dataclass(frozen=True, slots=True)
class _SweepGeneration:
    volumes: tuple[_SweptVolume, ...]


@dataclass(frozen=True, slots=True)
class _LayerAudit:
    requested: tuple[tuple[str, float], ...]
    achieved: tuple[tuple[str, float], ...]
    control_by_element_tag: dict[int, int]
    layer_by_element_tag: dict[int, int]


def _cad_symmetric_difference(gmsh, dimension: int, left, right, /) -> float:
    baseline = set(gmsh.model.getEntities())
    measure = 0.0
    for first, second in ((left, right), (right, left)):
        first_copy = gmsh.model.occ.copy([first])
        second_copy = gmsh.model.occ.copy([second])
        difference, _ = gmsh.model.occ.cut(first_copy, second_copy)
        measure += sum(
            gmsh.model.occ.getMass(dim, tag)
            for dim, tag in difference
            if dim == dimension
        )
    additions = sorted(set(gmsh.model.getEntities()) - baseline, reverse=True)
    if additions:
        gmsh.model.occ.remove(additions, recursive=True)
        gmsh.model.occ.synchronize()
    return float(measure)


def _certify_swept_volume(
    gmsh,
    volume: tuple[int, int],
    source_surface: int,
    target_surface: int,
    direction: np.ndarray,
    /,
) -> tuple[float, float]:
    baseline = set(gmsh.model.getEntities())
    translated = gmsh.model.occ.copy([(2, source_surface)])
    gmsh.model.occ.translate(translated, *direction.tolist())
    extruded = gmsh.model.occ.extrude(
        gmsh.model.occ.copy([(2, source_surface)]),
        *direction.tolist(),
    )
    gmsh.model.occ.synchronize()
    generated_volumes = tuple(entity for entity in extruded if entity[0] == 3)
    if len(translated) != 1 or len(generated_volumes) != 1:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Swept-layer CAD certification did not produce one translated cap and volume.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    face_difference = _cad_symmetric_difference(
        gmsh, 2, translated[0], (2, target_surface)
    )
    volume_difference = _cad_symmetric_difference(gmsh, 3, generated_volumes[0], volume)
    additions = sorted(set(gmsh.model.getEntities()) - baseline, reverse=True)
    if additions:
        gmsh.model.occ.remove(additions, recursive=True)
        gmsh.model.occ.synchronize()
    return face_difference, volume_difference


def _prepare_swept_geometry(gmsh, plan, shape, cad_entities, /):
    controls = plan.specification.layer_controls
    if not controls:
        return None
    source = _brep_model(plan.source)
    scale = max(float(np.ptp(np.asarray(source.mesh_vertices), axis=0).max()), 1.0)
    if cad_entities is None:
        volume_entities = tuple(gmsh.model.getEntities(3))
        if source.topology.num_solids != 1 or len(volume_entities) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Non-semantic swept meshing requires one source solid.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        face_to_surface = tuple(
            _resolve_entities(
                gmsh,
                source,
                shape,
                MeshingScope(
                    source.report.source_id,
                    source.report.source_revision,
                    MeshingEntityKind.GEOMETRY,
                    2,
                    f"{source.report.source_revision}:brep:2",
                    np.asarray((face,), dtype=np.int64),
                ),
            )[0]
            for face in range(source.report.num_faces)
        )
        solid_to_volume = (int(volume_entities[0][1]),)
    else:
        face_to_surface = cad_entities.face_to_surface
        solid_to_volume = cad_entities.solid_to_volume
    prepared = []
    for control_index, control in enumerate(controls):
        solid_ids = np.asarray(control.volume_scope.entity_ids, dtype=np.int64)
        source_faces = {
            int(value) for value in np.asarray(control.source_scope.entity_ids)
        }
        target_faces = {
            int(value) for value in np.asarray(control.target_scope.entity_ids)
        }
        thicknesses = np.asarray(control.schedule.thicknesses, dtype=np.float64)
        levels = np.concatenate(([0.0], np.cumsum(thicknesses)))
        for solid_value in solid_ids:
            solid = int(solid_value)
            source_candidates = source_faces & set(source.topology.solid_faces[solid])
            target_candidates = target_faces & set(source.topology.solid_faces[solid])
            if len(source_candidates) != 1 or len(target_candidates) != 1:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SPECIFICATION,
                    "Every controlled solid requires one exact source and target cap.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            source_face = source_candidates.pop()
            target_face = target_candidates.pop()
            source_surface = int(face_to_surface[source_face])
            target_surface = int(face_to_surface[target_face])
            if (
                source_surface == target_surface
                or gmsh.model.getType(2, source_surface) != "Plane"
                or gmsh.model.getType(2, target_surface) != "Plane"
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Swept layers require distinct planar source and target faces.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            origin = np.asarray(
                gmsh.model.occ.getCenterOfMass(2, source_surface), dtype=np.float64
            )
            target_center = np.asarray(
                gmsh.model.occ.getCenterOfMass(2, target_surface), dtype=np.float64
            )
            direction = target_center - origin
            length = float(np.linalg.norm(direction))
            tolerance = 1.0e-9 * max(scale, control.schedule.total_thickness)
            if (
                not np.isfinite(length)
                or length <= 0.0
                or abs(length - control.schedule.total_thickness) > tolerance
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SPECIFICATION,
                    "Swept-layer schedule total thickness does not equal the source-to-target translation.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            unit = direction / length
            _, parameters = gmsh.model.getClosestPoint(2, source_surface, origin.tolist())
            normal = np.asarray(
                gmsh.model.getNormal(source_surface, parameters), dtype=np.float64
            ).reshape(3)
            normal /= np.linalg.norm(normal)
            if abs(float(np.dot(normal, unit))) < 1.0 - 1.0e-10:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Swept-layer translation must be normal to its planar source face.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            volume_tag = int(solid_to_volume[solid])
            face_difference, volume_difference = _certify_swept_volume(
                gmsh,
                (3, volume_tag),
                source_surface,
                target_surface,
                direction,
            )
            face_area = float(gmsh.model.occ.getMass(2, source_surface))
            volume_measure = float(gmsh.model.occ.getMass(3, volume_tag))
            if face_difference > 1.0e-9 * max(
                face_area, scale**2
            ) or volume_difference > 1.0e-9 * max(volume_measure, scale**3):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Controlled volume is not the exact straight extrusion from source cap to target cap.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            lateral_faces = tuple(
                int(face)
                for face in source.topology.solid_faces[solid]
                if face not in (source_face, target_face)
            )
            prepared.append(
                _SweptVolume(
                    control_index,
                    control,
                    solid,
                    volume_tag,
                    source_face,
                    target_face,
                    source_surface,
                    target_surface,
                    lateral_faces,
                    tuple(int(face_to_surface[face]) for face in lateral_faces),
                    origin,
                    direction,
                    unit,
                    levels,
                    volume_difference / max(volume_measure, np.finfo(np.float64).tiny),
                )
            )
    by_solid = {value.solid_index: value for value in prepared}
    for value in prepared:
        for face in value.lateral_face_indices:
            adjacent = tuple(
                by_solid[owner]
                for owner in source.topology.face_solids[face]
                if owner != value.solid_index and owner in by_solid
            )
            for other in adjacent:
                if not np.allclose(
                    value.direction,
                    other.direction,
                    rtol=0.0,
                    atol=1.0e-9 * scale,
                ):
                    raise MeshingFailure(
                        MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                        "A swept cohort has inconsistent lateral translation vectors.",
                        stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                    )
    for value in prepared:
        transform = np.eye(4)
        transform[:3, 3] = value.direction
        gmsh.model.mesh.setPeriodic(
            2,
            [value.target_surface],
            [value.source_surface],
            transform.reshape(-1).tolist(),
        )
    return _SweepGeneration(tuple(prepared))


def _entity_linear_triangles(gmsh, surface: int, /) -> np.ndarray:
    blocks = []
    element_types, _, node_blocks = gmsh.model.mesh.getElements(2, surface)
    for element_type, node_values in zip(element_types, node_blocks, strict=True):
        name, _, order, count, _, corners = gmsh.model.mesh.getElementProperties(
            int(element_type)
        )
        if name.split()[0] != "Triangle" or int(order) != 1 or int(corners) != 3:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Swept source caps require complete linear triangle meshes.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        blocks.append(np.asarray(node_values, dtype=np.int64).reshape((-1, int(count))))
    if not blocks:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Swept source cap generated no triangles.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    return np.concatenate(blocks)


def _matching_lateral_surface(
    gmsh, candidates: tuple[int, ...], point: np.ndarray, tolerance: float, /
) -> int:
    matches = []
    for surface in candidates:
        closest, _ = gmsh.model.getClosestPoint(2, surface, point.tolist())
        closest_point = np.asarray(closest, dtype=np.float64).reshape((-1, 3))
        if (
            closest_point.shape == (1, 3)
            and np.linalg.norm(closest_point[0] - point) <= tolerance
            and gmsh.model.isInside(2, surface, point.tolist()) == 1
        ):
            matches.append(surface)
    if len(matches) != 1:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "An extruded source edge does not map uniquely to one lateral CAD surface.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    return matches[0]


def _install_swept_cells(gmsh, sweep: _SweepGeneration, /) -> None:
    for value in sweep.volumes:
        gmsh.model.mesh.removeElements(3, value.volume_tag)
    for surface in sorted(
        {surface for value in sweep.volumes for surface in value.lateral_surfaces}
    ):
        gmsh.model.mesh.removeElements(2, surface)
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coordinates = np.asarray(node_coordinates, dtype=np.float64).reshape((-1, 3))
    order = np.argsort(node_tags, kind="stable")
    node_tags = node_tags[order]
    node_coordinates = node_coordinates[order]
    coordinates = {
        int(tag): point for tag, point in zip(node_tags, node_coordinates, strict=True)
    }
    scale = max(float(np.ptp(node_coordinates, axis=0).max()), 1.0)
    tolerance = 1.0e-9 * scale
    coordinate_nodes = {
        tuple(np.rint(point / tolerance).astype(np.int64)): int(tag)
        for tag, point in zip(node_tags, node_coordinates, strict=True)
    }
    next_node_tag = int(np.max(node_tags, initial=0)) + 1
    new_nodes: dict[int, list[tuple[int, np.ndarray]]] = {}
    prism_blocks: dict[int, list[np.ndarray]] = {}
    quad_blocks: dict[int, dict[tuple[int, ...], np.ndarray]] = {}
    for value in sweep.volumes:
        triangles = _entity_linear_triangles(gmsh, value.source_surface)
        actual_master, slave_nodes, master_nodes, affine = (
            gmsh.model.mesh.getPeriodicNodes(
                2, value.target_surface, includeHighOrderNodes=True
            )
        )
        transform = np.eye(4)
        transform[:3, 3] = value.direction
        if int(actual_master) != value.source_surface or not np.allclose(
            np.asarray(affine, dtype=np.float64).reshape((4, 4)),
            transform,
            rtol=0.0,
            atol=tolerance,
        ):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Gmsh did not preserve the exact source-to-target cap translation.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        target_by_source = {
            int(master): int(slave)
            for slave, master in zip(slave_nodes, master_nodes, strict=True)
        }
        source_nodes = np.unique(triangles)
        if any(int(node) not in target_by_source for node in source_nodes):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Target cap is not a complete translated copy of the source triangle mesh.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        level_nodes: dict[int, tuple[int, ...]] = {}
        for source_node_value in source_nodes:
            source_node = int(source_node_value)
            route = [source_node]
            for distance in value.levels[1:-1]:
                point = coordinates[source_node] + value.unit * float(distance)
                key = tuple(np.rint(point / tolerance).astype(np.int64))
                node = coordinate_nodes.get(key)
                if node is None:
                    node = next_node_tag
                    next_node_tag += 1
                    coordinate_nodes[key] = node
                    coordinates[node] = point
                    new_nodes.setdefault(value.volume_tag, []).append((node, point))
                route.append(node)
            route.append(target_by_source[source_node])
            level_nodes[source_node] = tuple(route)
        edges = np.concatenate(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            )
        )
        edge_keys, edge_counts = np.unique(
            np.sort(edges, axis=1), axis=0, return_counts=True
        )
        boundary_edges = edge_keys[edge_counts == 1]
        for triangle in triangles:
            oriented = np.asarray(triangle, dtype=np.int64).copy()
            first, second, third = (coordinates[int(node)] for node in oriented)
            if np.dot(np.cross(second - first, third - first), value.direction) < 0.0:
                oriented[1], oriented[2] = oriented[2], oriented[1]
            for layer in range(value.control.schedule.layer_count):
                bottom = np.asarray(
                    [level_nodes[int(node)][layer] for node in oriented],
                    dtype=np.int64,
                )
                top = np.asarray(
                    [level_nodes[int(node)][layer + 1] for node in oriented],
                    dtype=np.int64,
                )
                prism_blocks.setdefault(value.volume_tag, []).append(
                    np.concatenate((bottom, top))
                )
        for edge in boundary_edges:
            first, second = (int(node) for node in edge)
            for layer in range(value.control.schedule.layer_count):
                quad = np.asarray(
                    (
                        level_nodes[first][layer],
                        level_nodes[second][layer],
                        level_nodes[second][layer + 1],
                        level_nodes[first][layer + 1],
                    ),
                    dtype=np.int64,
                )
                center = np.mean(
                    np.asarray([coordinates[int(node)] for node in quad]), axis=0
                )
                surface = _matching_lateral_surface(
                    gmsh, value.lateral_surfaces, center, tolerance
                )
                key = tuple(sorted(int(node) for node in quad))
                existing = quad_blocks.setdefault(surface, {}).get(key)
                if existing is not None and set(existing) != set(quad):
                    raise MeshingFailure(
                        MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        "A swept cohort produced inconsistent shared quad incidence.",
                        stage=MeshingStageKind.LAYER_GENERATION.value,
                    )
                quad_blocks[surface][key] = quad
    for volume_tag, entries in new_nodes.items():
        tags = np.asarray([tag for tag, _ in entries], dtype=np.int64)
        values = np.asarray([point for _, point in entries], dtype=np.float64)
        gmsh.model.mesh.addNodes(3, volume_tag, tags, values.reshape(-1))
    prism_type = gmsh.model.mesh.getElementType("Prism", 1)
    for volume_tag, prisms in prism_blocks.items():
        gmsh.model.mesh.addElementsByType(
            volume_tag,
            prism_type,
            [],
            np.asarray(prisms, dtype=np.int64).reshape(-1),
        )
    quadrilateral_type = gmsh.model.mesh.getElementType("Quadrangle", 1)
    for surface, quads in quad_blocks.items():
        gmsh.model.mesh.addElementsByType(
            surface,
            quadrilateral_type,
            [],
            np.asarray(tuple(quads.values()), dtype=np.int64).reshape(-1),
        )


def _audit_layers(sweep, rows, node_tags, points, /) -> _LayerAudit:
    if sweep is None:
        return _LayerAudit((), (), {}, {})
    volume_map = {value.volume_tag: value for value in sweep.volumes}
    prism_rows = tuple(row for row in rows if row.cell_kind == "prism")
    control_by_tag = {}
    layer_by_tag = {}
    evaluations: dict[int, list] = {}
    for block in rows:
        controlled = np.asarray(
            [int(tag) in volume_map for tag in block.entity_tags], dtype=np.bool_
        )
        if np.any(controlled) and block.cell_kind != "prism":
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A controlled swept volume contains a non-prism cell.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        if block.cell_kind == "prism" and np.any(~controlled):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A prism cell was generated outside the controlled swept volumes.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
    for value in sweep.volumes:
        selected_vertices = []
        selected_tags = []
        for block in prism_rows:
            selected = block.entity_tags == value.volume_tag
            if np.any(selected):
                selected_vertices.append(block.vertices[selected, : block.corner_count])
                selected_tags.append(block.tags[selected])
        if not selected_vertices:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A controlled swept volume contains no generated prisms.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        tags = np.concatenate(selected_tags)
        vertices = _local_connectivity(node_tags, np.concatenate(selected_vertices))
        corners = points[vertices]
        lower = np.mean((corners[:, :3] - value.origin) @ value.unit, axis=1)
        intervals = np.argmin(np.abs(lower[:, None] - value.levels[None, :-1]), axis=1)
        evaluation = evaluate_swept_layer_quality(
            points,
            vertices,
            intervals,
            value.origin,
            value.unit,
            value.control.schedule.thicknesses,
        )
        tolerance = 1.0e-9 * max(float(value.levels[-1]), 1.0)
        if (
            not evaluation.valid
            or evaluation.maximum_thickness_residual > tolerance
            or evaluation.maximum_alignment_residual > 1.0e-10
            or evaluation.maximum_interface_residual > tolerance
            or np.unique(intervals).size != value.control.schedule.layer_count
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Generated prisms do not realize the exact requested thickness, alignment, and interface levels.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        evaluations.setdefault(value.control_index, []).append(evaluation)
        for tag, layer in zip(tags, intervals, strict=True):
            control_by_tag[int(tag)] = value.control_index
            layer_by_tag[int(tag)] = int(layer)
    requested = []
    achieved = []
    control_map = {value.control_index: value.control for value in sweep.volumes}
    controls = tuple(control_map[index] for index in sorted(control_map))
    for control_index, control in zip(sorted(control_map), controls, strict=True):
        key = f"layer:{control.control_id}"
        requested.extend(
            (
                (f"{key}:layer_count", float(control.schedule.layer_count)),
                (f"{key}:total_thickness", control.schedule.total_thickness),
                *(
                    (f"{key}:thickness:{layer}", thickness)
                    for layer, thickness in enumerate(control.schedule.thicknesses)
                ),
            )
        )
        local = evaluations[control_index]
        measured = np.mean(
            np.asarray(
                [np.asarray(value.measured_thicknesses) for value in local],
                dtype=np.float64,
            ),
            axis=0,
        )
        growth = (
            measured[1:] / measured[:-1]
            if measured.size > 1
            else np.empty((0,), dtype=np.float64)
        )
        achieved.extend(
            (
                (f"{key}:layer_count", float(measured.size)),
                (f"{key}:total_thickness", float(np.sum(measured))),
                *(
                    (f"{key}:thickness:{layer}", float(thickness))
                    for layer, thickness in enumerate(measured)
                ),
                *(
                    (f"{key}:growth:{layer}", float(ratio))
                    for layer, ratio in enumerate(growth, start=1)
                ),
                (
                    f"{key}:maximum_thickness_residual",
                    max(value.maximum_thickness_residual for value in local),
                ),
                (
                    f"{key}:maximum_alignment_residual",
                    max(value.maximum_alignment_residual for value in local),
                ),
                (
                    f"{key}:maximum_interface_residual",
                    max(value.maximum_interface_residual for value in local),
                ),
                (
                    f"{key}:relative_cad_symmetric_difference",
                    max(
                        value.relative_cad_difference
                        for value in sweep.volumes
                        if value.control_index == control_index
                    ),
                ),
            )
        )
    if len(controls) == 1:
        control = controls[0]
        measured = np.asarray(evaluations[0][0].measured_thicknesses, dtype=np.float64)
        achieved.extend(
            (
                ("layer_count", float(measured.size)),
                ("first_layer_thickness", float(measured[0])),
                (
                    "layer_growth_rate",
                    float(np.max(measured[1:] / measured[:-1]))
                    if measured.size > 1
                    else 1.0,
                ),
                (
                    "layer_interface_maximum_residual",
                    max(value.maximum_interface_residual for value in evaluations[0]),
                ),
            )
        )
        requested.extend(
            (
                ("layer_count", float(control.schedule.layer_count)),
                ("total_layer_thickness", control.schedule.total_thickness),
            )
        )
    return _LayerAudit(
        tuple(requested),
        tuple(achieved),
        control_by_tag,
        layer_by_tag,
    )


def _layer_attributes(
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    audit: _LayerAudit,
    /,
) -> tuple[MeshAttribute, ...]:
    if not audit.layer_by_element_tag:
        return ()
    cell_ids = []
    control_indices = []
    layer_indices = []
    rows_by_name = {row.block_name: row for row in rows}
    for block in mesh.blocks:
        if block.cell_kind != "prism":
            continue
        source_rows = rows_by_name[block.name]
        tags = source_rows.tags[row_orders[block.name]]
        cell_ids.extend(int(value) for value in np.asarray(block.global_ids))
        control_indices.extend(audit.control_by_element_tag[int(tag)] for tag in tags)
        layer_indices.extend(audit.layer_by_element_tag[int(tag)] for tag in tags)
    entity_set = mesh.entity_set(3)
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        3,
        entity_set.entity_set_id,
        np.asarray(cell_ids, dtype=np.int64),
    )
    return (
        MeshAttribute(
            "layer_index",
            MeshAttributeRole.MARKER,
            scope,
            np.asarray(layer_indices, dtype=np.int32),
        ),
        MeshAttribute(
            "layer_control_index",
            MeshAttributeRole.MARKER,
            scope,
            np.asarray(control_indices, dtype=np.int32),
        ),
    )


def _size_values(specification: SurfaceMeshingSpec | VolumeMeshingSpec, /):
    top_scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
    whole = tuple(
        control
        for control in specification.size_controls
        if not isinstance(control, ProximitySizeControl)
        and control.scope.scope_id == top_scope.scope_id
    )
    uniform = tuple(
        control for control in whole if isinstance(control, UniformSizeControl)
    )
    hard = tuple(
        control for control in whole if control.strength is SizeControlStrength.HARD
    )
    minimum = max(
        (control.minimum_size for control in hard if control.minimum_size is not None),
        default=None,
    )
    maximum = min(
        (control.maximum_size for control in hard if control.maximum_size is not None),
        default=None,
    )
    if not uniform:
        target = None
    elif specification.size_combination is SizeCombinationPolicy.REJECT_HARD_CONFLICTS:
        target = min(control.target_size for control in uniform)
    else:
        hard_uniform = tuple(
            control for control in uniform if control.strength is SizeControlStrength.HARD
        )
        candidates = hard_uniform if hard_uniform else uniform
        priority = max(control.priority for control in candidates)
        target = next(
            control.target_size for control in candidates if control.priority == priority
        )
    if target is not None:
        if minimum is not None:
            target = max(target, minimum)
        if maximum is not None:
            target = min(target, maximum)
    curvature_angles = tuple(
        control.normal_angle
        for control in whole
        if isinstance(control, CurvatureSizeControl)
    )
    curvature_points = (
        0 if not curvature_angles else int(np.ceil(2.0 * np.pi / min(curvature_angles)))
    )
    return minimum, target, maximum, curvature_points


def _apply_uniform_size_fields(
    gmsh,
    source: BRepModel,
    shape,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    cad_entities: _CadEntityMap | None,
    outside_size: float,
    /,
) -> tuple[int, ...]:
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if not controls:
        return ()
    semantic_volume = isinstance(specification, VolumeMeshingSpec) and bool(
        specification.region_controls
    )
    if semantic_volume:
        if cad_entities is None:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Semantic solid size fields require resolved CAD volume identities.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        solid_ids = np.arange(source.topology.num_solids, dtype=np.int64)
        resolved, _ = resolve_size_controls(
            controls,
            np.zeros((solid_ids.size, 1), dtype=np.float64),
            solid_ids,
            SizeFieldDomain.EUCLIDEAN_VOLUME,
            combination=specification.size_combination,
        )
        values = np.asarray(resolved.values, dtype=np.float64)
        groups = tuple(
            (
                float(value),
                tuple(
                    cad_entities.solid_to_volume[int(index)]
                    for index in np.flatnonzero(values == value)
                ),
            )
            for value in np.unique(values)
        )
        dimension = 3
    else:
        top_scope = (
            specification.scope
            if isinstance(specification, SurfaceMeshingSpec)
            else specification.boundary_scope
        )
        if specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY:
            groups = [
                (
                    outside_size,
                    tuple(
                        tag
                        for _, tag in gmsh.model.getEntities(top_scope.entity_dimension)
                    ),
                    top_scope.entity_dimension,
                )
            ]
        else:
            groups = []
            for control in controls:
                dimension = control.scope.entity_dimension
                tags = (
                    tuple(tag for _, tag in gmsh.model.getEntities(dimension))
                    if control.scope.scope_id == top_scope.scope_id
                    else _resolve_entities(gmsh, source, shape, control.scope)
                )
                groups.append((control.target_size, tags, dimension))
    fields = []
    list_names = {
        0: "PointsList",
        1: "CurvesList",
        2: "SurfacesList",
        3: "VolumesList",
    }
    if semantic_volume:
        grouped = tuple((value, tags, dimension) for value, tags in groups)
    else:
        grouped = tuple(groups)
    for value, tags, entity_dimension in grouped:
        field = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumber(field, "VIn", value)
        gmsh.model.mesh.field.setNumber(field, "VOut", outside_size)
        gmsh.model.mesh.field.setNumbers(field, list_names[entity_dimension], list(tags))
        gmsh.model.mesh.field.setNumber(field, "IncludeBoundary", 1)
        fields.append(field)
    background = fields[0]
    if len(fields) > 1:
        background = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(background, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(background)
    return tuple(fields)


def _boundary_association(
    source: BRepModel,
    boundary: CellMesh,
    tolerance_factor: float,
    /,
) -> tuple[GeometryAssociation, tuple[MeshZone, ...], MeshAttribute]:
    points = np.asarray(boundary.coordinates, dtype=np.float64)
    centroids = np.concatenate(
        [
            np.mean(points[np.asarray(block.vertices, dtype=np.int32)], axis=1)
            for block in boundary.blocks
        ]
    )
    query_mesh = TriangleMesh(
        source.mesh_vertices,
        source.mesh_faces,
        source_id=f"{source.report.source_id}:association-query",
    )
    query = query_mesh.query_index().query(jnp.asarray(centroids))
    triangle_ids = np.asarray(query.face_index, dtype=np.int32)
    source_faces = np.asarray(source.triangle_face_ids, dtype=np.int32)[triangle_ids]
    residuals = np.asarray(query.distance, dtype=np.float64)
    tolerance = max(
        source.report.linear_deflection * float(tolerance_factor),
        256.0 * np.finfo(np.float64).eps,
    )
    resolved = residuals <= tolerance
    target_set = boundary.entity_set(2)
    source_ids = tuple(
        f"{source.report.source_revision}:face:{int(index)}" for index in source_faces
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        target_set.entity_set_id,
        target_set.entity_ids,
        source_ids,
        residuals,
        resolved=resolved,
        exact=False,
    )
    if not association.complete:
        failed = tuple(np.asarray(target_set.entity_ids)[~resolved])
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "Generated boundary faces could not be uniquely matched within tolerance.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            entity_ids=failed,
        )
    zones = []
    for face_id in np.unique(source_faces):
        selected = np.flatnonzero(source_faces == face_id)
        scope = MeshingScope(
            boundary.mesh_id,
            boundary.numeric_version,
            MeshingEntityKind.MESH,
            2,
            target_set.entity_set_id,
            np.asarray(target_set.entity_ids)[selected],
        )
        zones.append(MeshZone(f"brep-face-{int(face_id)}", MeshZoneRole.BOUNDARY, scope))
    all_scope = MeshingScope(
        boundary.mesh_id,
        boundary.numeric_version,
        MeshingEntityKind.MESH,
        2,
        target_set.entity_set_id,
        target_set.entity_ids,
    )
    attribute = MeshAttribute(
        "brep_face_index",
        MeshAttributeRole.GEOMETRY_CLASSIFICATION,
        all_scope,
        source_faces,
    )
    return association, tuple(zones), attribute


@dataclass(frozen=True, slots=True)
class _PlanarSurfaceEvidence:
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    associations: tuple[GeometryAssociation, ...]
    attributes: tuple[MeshAttribute, ...]
    cell_face_ids: np.ndarray
    mesh_edge_source: np.ndarray


def _curve_corner_rows(gmsh, geometry_order: int, /) -> tuple[np.ndarray, np.ndarray]:
    node_chunks = []
    entity_chunks = []
    for _, curve in sorted(gmsh.model.getEntities(1)):
        element_types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(1, curve)
        for element_type, tag_values, node_values in zip(
            element_types, tag_blocks, node_blocks, strict=True
        ):
            name, dimension, order, count, _, corners = (
                gmsh.model.mesh.getElementProperties(int(element_type))
            )
            tags = np.asarray(tag_values, dtype=np.int64)
            if (
                name.split()[0] != "Line"
                or int(dimension) != 1
                or int(order) != geometry_order
                or int(corners) != 2
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    f"Gmsh returned unsupported planar curve element {name!r}.",
                    stage=MeshingStageKind.CANONICALIZATION.value,
                )
            nodes = np.asarray(node_values, dtype=np.int64).reshape((-1, int(count)))
            node_chunks.append(nodes[:, :2])
            entity_chunks.append(np.full(tags.shape, curve, dtype=np.int64))
    if not node_chunks:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh returned no planar curve elements.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return np.concatenate(node_chunks), np.concatenate(entity_chunks)


def _polygon_edge_incidents(
    connectivity: PolygonalConnectivity, /
) -> tuple[tuple[int, ...], ...]:
    incidents = [[] for _ in np.asarray(connectivity.edges)]
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
    for cell_index, row in enumerate(cell_edges):
        for edge_index in row[valid[cell_index]]:
            incidents[int(edge_index)].append(cell_index)
    return tuple(tuple(values) for values in incidents)


def _planar_edge_patch_connected(
    connectivity: PolygonalConnectivity, edge_indices: np.ndarray, /
) -> bool:
    if edge_indices.size <= 1:
        return True
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    vertex_edges: dict[int, set[int]] = {}
    selected = {int(value) for value in edge_indices}
    for edge_index in selected:
        for vertex in edges[edge_index]:
            vertex_edges.setdefault(int(vertex), set()).add(edge_index)
    pending = [next(iter(selected))]
    visited = set()
    while pending:
        edge_index = pending.pop()
        if edge_index in visited:
            continue
        visited.add(edge_index)
        for vertex in edges[edge_index]:
            pending.extend(vertex_edges[int(vertex)] - visited)
    return visited == selected


def _planar_surface_evidence(
    gmsh,
    source: BRepModel,
    mesh: CellMesh,
    specification: SurfaceMeshingSpec,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    node_tags: np.ndarray,
    source_to_corner: np.ndarray,
    cad_entities: _CadEntityMap,
    geometry_order: int,
    /,
) -> _PlanarSurfaceEvidence:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Semantic planar meshing requires PolygonalConnectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    surface_to_face = {
        surface: face for face, surface in enumerate(cad_entities.face_to_surface)
    }
    rows_by_name = {row.block_name: row for row in rows}
    owner_chunks = []
    for block in mesh.blocks:
        row = rows_by_name[block.name]
        entity_tags = row.entity_tags[row_orders[block.name]]
        owner_chunks.append(
            np.asarray(
                tuple(surface_to_face.get(int(tag), -1) for tag in entity_tags),
                dtype=np.int32,
            )
        )
    cell_face_ids = np.concatenate(owner_chunks)
    cell_entity_set = mesh.entity_set(2)
    cell_ids = np.asarray(cell_entity_set.entity_ids, dtype=np.int64)
    if (
        cell_face_ids.shape != cell_ids.shape
        or np.any(cell_face_ids < 0)
        or not np.array_equal(
            cell_ids,
            np.concatenate(
                tuple(
                    np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks
                )
            ),
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CANONICALIZATION_FAILED,
            "Canonical planar cells lost their source-face ownership.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )

    curve_nodes, curve_tags = _curve_corner_rows(gmsh, geometry_order)
    local_nodes = _local_connectivity(node_tags, curve_nodes)
    corners = source_to_corner[local_nodes]
    if np.any(corners < 0):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "A source curve corner is absent from the canonical planar mesh.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    edge_rows = np.asarray(connectivity.edges, dtype=np.int32)
    edge_lookup = {
        tuple(sorted(int(value) for value in edge)): index
        for index, edge in enumerate(edge_rows)
    }
    curve_to_edge = {curve: edge for edge, curve in enumerate(cad_entities.edge_to_curve)}
    mesh_edge_source = np.full((edge_rows.shape[0],), -1, dtype=np.int32)
    for values, curve in zip(corners, curve_tags, strict=True):
        mesh_edge = edge_lookup.get(tuple(sorted(int(value) for value in values)))
        source_edge = curve_to_edge.get(int(curve))
        if mesh_edge is None or source_edge is None or mesh_edge_source[mesh_edge] >= 0:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "A Gmsh curve element does not map uniquely to one canonical mesh edge.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        mesh_edge_source[mesh_edge] = source_edge
    if {int(value) for value in mesh_edge_source if value >= 0} != set(
        range(source.report.num_edges)
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Generated planar edges do not cover every source BRep edge.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    incidents = _polygon_edge_incidents(connectivity)
    mapped = np.flatnonzero(mesh_edge_source >= 0)
    for mesh_edge in mapped:
        source_edge = int(mesh_edge_source[mesh_edge])
        expected = set(source.topology.edge_faces[source_edge])
        actual = {int(cell_face_ids[cell]) for cell in incidents[int(mesh_edge)]}
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Canonical planar edge adjacency differs from source BRep incidence.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
    expected_boundary = np.zeros((edge_rows.shape[0],), dtype=np.bool_)
    expected_boundary[mapped] = np.asarray(
        [
            len(source.topology.edge_faces[int(mesh_edge_source[index])]) == 1
            for index in mapped
        ]
    )
    if not np.array_equal(
        np.asarray(connectivity.boundary_edges, dtype=np.bool_), expected_boundary
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Planar mesh boundary is not exactly the singly incident source CAD edges.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )

    face_association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        cell_entity_set.entity_set_id,
        cell_ids,
        tuple(
            f"{source.report.source_revision}:face:{int(face)}" for face in cell_face_ids
        ),
        np.zeros(cell_ids.shape, dtype=np.float64),
        exact=True,
    )
    edge_entity_set = mesh.entity_set(1)
    edge_ids = np.asarray(edge_entity_set.entity_ids, dtype=np.int64)
    edge_association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        edge_entity_set.entity_set_id,
        edge_ids[mapped],
        tuple(
            f"{source.report.source_revision}:edge:{int(mesh_edge_source[index])}"
            for index in mapped
        ),
        np.zeros(mapped.shape, dtype=np.float64),
        exact=True,
    )

    face_regions = np.empty((source.report.num_faces,), dtype=object)
    face_regions[:] = None
    zones = []
    for control in specification.region_controls:
        source_faces = np.asarray(control.scope.entity_ids, dtype=np.int32)
        face_regions[source_faces] = control.region_name
        selected = np.isin(cell_face_ids, source_faces)
        if not np.any(selected):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Region {control.region_name!r} has no generated planar cells.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            cell_entity_set.entity_set_id,
            cell_ids[selected],
        )
        zones.append(
            MeshZone(
                control.region_name,
                MeshZoneRole.REGION,
                scope,
                material_id=control.material_id,
                region_role=control.role,
            )
        )
    if any(value is None for value in face_regions):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Planar cell ownership is not an exhaustive region partition.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    region_by_cell = face_regions[cell_face_ids]
    zone_by_name = {zone.name: zone for zone in zones}
    patches = []
    claimed: set[int] = set()
    for control in specification.patch_controls:
        source_edges = np.asarray(control.scope.entity_ids, dtype=np.int32)
        selected = np.flatnonzero(np.isin(mesh_edge_source, source_edges))
        if {int(value) for value in mesh_edge_source[selected]} != {
            int(value) for value in source_edges
        }:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated patch {control.name!r} does not cover its exact source scope.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        for mesh_edge in selected:
            adjacent = incidents[int(mesh_edge)]
            actual = tuple(sorted(str(region_by_cell[cell]) for cell in adjacent))
            if (
                len(adjacent) not in (1, 2)
                or len(set(actual)) != len(actual)
                or actual != control.adjacent_region_names
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Generated patch {control.name!r} has incorrect planar region adjacency.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                    entity_ids=(int(edge_ids[int(mesh_edge)]),),
                )
        if not selected.size:
            if control.required:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Required patch {control.name!r} is absent.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            continue
        claimed.update(int(value) for value in selected)
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            1,
            edge_entity_set.entity_set_id,
            edge_ids[selected],
        )
        patches.append(
            MeshPatch(
                control.name,
                scope,
                connected=_planar_edge_patch_connected(connectivity, selected),
                adjacent_zone_ids=tuple(
                    zone_by_name[name].zone_id for name in control.adjacent_region_names
                ),
            )
        )
    for mesh_edge, adjacent in enumerate(incidents):
        if len(adjacent) != 2:
            continue
        regions = {
            str(region_by_cell[adjacent[0]]),
            str(region_by_cell[adjacent[1]]),
        }
        if len(regions) == 2 and mesh_edge not in claimed:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated planar inter-region edge {mesh_edge} is undeclared.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                entity_ids=(int(edge_ids[mesh_edge]),),
            )

    face_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        2,
        cell_entity_set.entity_set_id,
        cell_ids,
    )
    edge_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        edge_entity_set.entity_set_id,
        edge_ids[mapped],
    )
    attributes = (
        MeshAttribute(
            "brep_face_index",
            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
            face_scope,
            cell_face_ids,
        ),
        MeshAttribute(
            "brep_edge_index",
            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
            edge_scope,
            mesh_edge_source[mapped],
        ),
    )
    return _PlanarSurfaceEvidence(
        tuple(zones),
        tuple(patches),
        (face_association, edge_association),
        attributes,
        cell_face_ids,
        mesh_edge_source,
    )


def _canonical_cell_solid_ids(
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    cad_entities: _CadEntityMap,
    /,
) -> np.ndarray:
    volume_to_solid = {
        volume: solid for solid, volume in enumerate(cad_entities.solid_to_volume)
    }
    rows_by_name = {row.block_name: row for row in rows}
    owner_chunks = []
    for block in mesh.blocks:
        row = rows_by_name[block.name]
        entity_tags = row.entity_tags[row_orders[block.name]]
        owner_chunks.append(
            np.asarray(
                tuple(volume_to_solid.get(int(tag), -1) for tag in entity_tags),
                dtype=np.int32,
            )
        )
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    if not np.array_equal(cell_ids, np.asarray(mesh.entity_set(3).entity_ids)):
        raise MeshingFailure(
            MeshingFailureCategory.CANONICALIZATION_FAILED,
            "Canonical cell entity ordering does not match canonical mesh blocks.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    owners = np.concatenate(owner_chunks)
    if np.any(owners < 0):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "A generated top cell has unknown source-solid ownership.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return owners


@dataclass(frozen=True, slots=True)
class _SemanticSurfaceEvidence:
    boundary: SurfaceModel
    association: GeometryAssociation
    zones: tuple[MeshZone, ...]
    attribute: MeshAttribute
    mesh_face_source: np.ndarray


def _connectivity_face_rows(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity, /
) -> tuple[np.ndarray, ...]:
    if isinstance(connectivity, TetrahedralConnectivity):
        return tuple(np.asarray(connectivity.faces, dtype=np.int32))
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int32)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
    return tuple(values[int(start) : int(stop)] for start, stop in pairwise(offsets))


def _connectivity_face_incidents(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity, /
) -> tuple[tuple[int, ...], ...]:
    if isinstance(connectivity, PolyhedralConnectivity):
        owner = np.asarray(connectivity.face_owner, dtype=np.int32)
        neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32)
        return tuple(
            (int(first),) if int(second) < 0 else (int(first), int(second))
            for first, second in zip(owner, neighbor, strict=True)
        )
    incidents = [[] for _ in np.asarray(connectivity.faces)]
    for cell_index, face_row in enumerate(
        np.asarray(connectivity.cell_faces, dtype=np.int32)
    ):
        for face_index in face_row:
            incidents[int(face_index)].append(cell_index)
    return tuple(tuple(row) for row in incidents)


def _connectivity_face_edge_rows(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity, /
) -> tuple[np.ndarray, ...]:
    if isinstance(connectivity, TetrahedralConnectivity):
        return tuple(np.asarray(connectivity.face_edges, dtype=np.int32))
    offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int32)
    values = np.asarray(connectivity.face_edge_values, dtype=np.int32)
    return tuple(values[int(start) : int(stop)] for start, stop in pairwise(offsets))


def _semantic_surface_evidence(
    gmsh,
    source: BRepModel,
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    node_tags: np.ndarray,
    source_to_corner: np.ndarray,
    cell_solid_ids: np.ndarray,
    cad_entities: _CadEntityMap,
    plan_id: str,
    /,
) -> _SemanticSurfaceEvidence:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, (TetrahedralConnectivity, PolyhedralConnectivity)):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Semantic BRep volume meshing requires tetrahedral or mixed polyhedral connectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    faces = _connectivity_face_rows(connectivity)
    face_lookup = {
        tuple(sorted(int(value) for value in face)): index
        for index, face in enumerate(faces)
    }
    surface_to_face = {
        surface: face for face, surface in enumerate(cad_entities.face_to_surface)
    }
    face_source = np.full((len(faces),), -1, dtype=np.int32)
    for block in rows:
        if block.cell_kind not in ("triangle", "quadrilateral"):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Semantic CAD surfaces require triangle or swept-quad elements.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        local_nodes = _local_connectivity(
            node_tags, block.vertices[:, : block.corner_count]
        )
        corners = source_to_corner[local_nodes]
        if np.any(corners < 0):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "A CAD surface corner is absent from the canonical volume mesh.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        for values, surface_tag in zip(corners, block.entity_tags, strict=True):
            face_index = face_lookup.get(tuple(sorted(int(value) for value in values)))
            source_face = surface_to_face.get(int(surface_tag))
            if face_index is None or source_face is None or face_source[face_index] >= 0:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    "A Gmsh CAD surface element does not map uniquely to one canonical mesh face.",
                    stage=MeshingStageKind.CANONICALIZATION.value,
                )
            face_source[face_index] = source_face

    incidents = _connectivity_face_incidents(connectivity)
    boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    mapped = np.flatnonzero(face_source >= 0)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    for face_index in mapped:
        source_face = int(face_source[face_index])
        expected = set(source.topology.face_solids[source_face])
        actual = {int(cell_solid_ids[cell]) for cell in incidents[face_index]}
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Canonical CAD face adjacency does not match source solid incidence.",
                stage=MeshingStageKind.CANONICALIZATION.value,
                entity_ids=(int(face_ids[face_index]),),
            )
    expected_exterior = face_source >= 0
    expected_exterior[mapped] = np.asarray(
        [
            len(source.topology.face_solids[int(face_source[index])]) == 1
            for index in mapped
        ]
    )
    if not np.array_equal(boundary_mask, expected_exterior):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Exterior mesh faces are not exactly the singly incident source CAD surfaces.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )

    face_entity_set = mesh.entity_set(2)
    exterior = np.flatnonzero(boundary_mask)
    boundary_triangles = []
    boundary_sources = []
    for face_index in exterior:
        face = faces[int(face_index)]
        for local in range(1, face.size - 1):
            boundary_triangles.append(
                (int(face[0]), int(face[local]), int(face[local + 1]))
            )
            boundary_sources.append(int(face_source[face_index]))
    boundary_triangles_array = np.asarray(boundary_triangles, dtype=np.int32)
    if all(faces[int(index)].size == 3 for index in exterior):
        boundary_order = np.arange(exterior.size, dtype=np.int64)
        boundary_ids = face_ids[exterior]
    else:
        boundary_keys = np.sort(boundary_triangles_array, axis=1)
        boundary_order = np.lexsort(
            tuple(
                boundary_keys[:, column]
                for column in range(boundary_keys.shape[1] - 1, -1, -1)
            )
        )
        boundary_ids = np.arange(boundary_order.size, dtype=np.int64)
    ordered_sources = np.asarray(boundary_sources, dtype=np.int32)[boundary_order]
    boundary_metadata = SurfaceMetadata(
        source_id=source.report.source_id,
        source_revision=source.report.source_revision,
        coordinate_contract=source.coordinate_contract,
        provenance=("gmsh-occ", plan_id),
        cell_tags=tuple(f"brep-face:{int(value)}" for value in ordered_sources),
    )
    boundary = SurfaceModel.from_triangles(
        mesh.coordinates,
        boundary_triangles_array[boundary_order],
        boundary_metadata,
        vertex_global_ids=mesh.vertex_global_ids,
        cell_global_ids=boundary_ids,
        numeric_version=mesh.numeric_version,
        repair_orientation=True,
        orient_closed_outward=True,
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        face_entity_set.entity_set_id,
        face_ids[mapped],
        tuple(
            f"{source.report.source_revision}:face:{int(face_source[index])}"
            for index in mapped
        ),
        np.zeros((mapped.size,), dtype=np.float64),
        exact=True,
    )
    zones = []
    for source_face in np.unique(face_source[exterior]):
        selected = exterior[face_source[exterior] == source_face]
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            face_entity_set.entity_set_id,
            face_ids[selected],
        )
        zones.append(
            MeshZone(f"brep-face-{int(source_face)}", MeshZoneRole.BOUNDARY, scope)
        )
    attribute_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        2,
        face_entity_set.entity_set_id,
        face_ids[mapped],
    )
    attribute = MeshAttribute(
        "brep_face_index",
        MeshAttributeRole.GEOMETRY_CLASSIFICATION,
        attribute_scope,
        face_source[mapped],
    )
    return _SemanticSurfaceEvidence(
        boundary,
        association,
        tuple(zones),
        attribute,
        face_source,
    )


def _audit_swept_interfaces(
    mesh: CellMesh,
    source: BRepModel,
    cell_solid_ids: np.ndarray,
    mesh_face_source: np.ndarray,
    sweep: _SweepGeneration | None,
    /,
) -> tuple[tuple[str, float], ...]:
    if sweep is None or {block.cell_kind for block in mesh.blocks} != {
        "prism",
        "tetrahedron",
    }:
        return ()
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Mixed swept output requires canonical PolyhedralConnectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    faces = _connectivity_face_rows(connectivity)
    incidents = _connectivity_face_incidents(connectivity)
    cell_kinds = np.concatenate(
        tuple(
            np.full((block.cell_count,), block.cell_kind, dtype=object)
            for block in mesh.blocks
        )
    )
    controlled = {value.solid_index for value in sweep.volumes}
    expected_caps = {
        face
        for value in sweep.volumes
        for face in (value.source_face_index, value.target_face_index)
        if any(owner not in controlled for owner in source.topology.face_solids[face])
    }
    observed_caps = set()
    for face_index, adjacent in enumerate(incidents):
        kinds = {str(cell_kinds[cell]) for cell in adjacent}
        source_face = int(mesh_face_source[face_index])
        if kinds == {"prism", "tetrahedron"}:
            if len(faces[face_index]) != 3 or source_face not in expected_caps:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Prism/tetrahedron cells may meet only on a triangular swept cap.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            observed_caps.add(source_face)
        if len(faces[face_index]) == 4 and "tetrahedron" in kinds:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A swept quad curtain adjoins a tetrahedral cell.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
    if observed_caps != expected_caps:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Swept cap triangles do not form the complete conforming prism/tetrahedron interface.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    return (("layer_interface_compliance", 1.0),)


def _patch_is_connected(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity,
    face_indices: np.ndarray,
    /,
) -> bool:
    if face_indices.size <= 1:
        return True
    face_edges = _connectivity_face_edge_rows(connectivity)
    edge_faces: dict[int, set[int]] = {}
    for face_index in face_indices:
        for edge_index in face_edges[int(face_index)]:
            edge_faces.setdefault(int(edge_index), set()).add(int(face_index))
    pending = [int(face_indices[0])]
    visited = set()
    while pending:
        face_index = pending.pop()
        if face_index in visited:
            continue
        visited.add(face_index)
        for edge_index in face_edges[face_index]:
            pending.extend(edge_faces[int(edge_index)] - visited)
    return len(visited) == face_indices.size


def _region_evidence(
    source: BRepModel,
    mesh: CellMesh,
    specification: VolumeMeshingSpec,
    cell_solid_ids: np.ndarray,
    mesh_face_source: np.ndarray,
    /,
) -> tuple[tuple[MeshZone, ...], tuple[MeshPatch, ...]]:
    cell_entity_set = mesh.entity_set(3)
    cell_ids = np.asarray(cell_entity_set.entity_ids, dtype=np.int64)
    solid_regions = np.empty((source.topology.num_solids,), dtype=object)
    solid_regions[:] = None
    zones = []
    for control in specification.region_controls:
        solid_ids = np.asarray(control.scope.entity_ids, dtype=np.int64)
        solid_regions[solid_ids] = control.region_name
        selected = np.isin(cell_solid_ids, solid_ids)
        if not np.any(selected):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Region {control.region_name!r} has no generated cells.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            3,
            cell_entity_set.entity_set_id,
            cell_ids[selected],
        )
        zones.append(
            MeshZone(
                control.region_name,
                MeshZoneRole.REGION,
                scope,
                material_id=control.material_id,
                region_role=control.role,
            )
        )
    if any(value is None for value in solid_regions):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Generated cell ownership does not resolve to an exhaustive region partition.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    region_by_cell = solid_regions[cell_solid_ids]
    zone_by_name = {zone.name: zone for zone in zones}
    connectivity = mesh.connectivity
    if not isinstance(connectivity, (TetrahedralConnectivity, PolyhedralConnectivity)):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Semantic region patches require tetrahedral or mixed polyhedral connectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    face_source = np.asarray(mesh_face_source, dtype=np.int32)
    face_rows = _connectivity_face_rows(connectivity)
    if face_source.shape != (len(face_rows),):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "CAD face evidence does not align with canonical mesh faces.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    incidents = _connectivity_face_incidents(connectivity)
    face_entity_set = mesh.entity_set(2)
    face_ids = np.asarray(face_entity_set.entity_ids, dtype=np.int64)
    claimed: set[int] = set()
    patches = []
    for control in specification.patch_controls:
        selected = np.flatnonzero(
            np.isin(
                face_source,
                np.asarray(control.scope.entity_ids, dtype=np.int32),
            )
        )
        requested_source_faces = {
            int(value) for value in np.asarray(control.scope.entity_ids)
        }
        mapped_source_faces = {int(value) for value in face_source[selected]}
        if selected.size and mapped_source_faces != requested_source_faces:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated patch {control.name!r} does not cover its exact source scope.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        valid = []
        for face_index in selected:
            adjacent = incidents[int(face_index)]
            actual = tuple(sorted(str(region_by_cell[cell]) for cell in adjacent))
            if (
                len(adjacent) not in (1, 2)
                or len(set(actual)) != len(actual)
                or actual != control.adjacent_region_names
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Generated patch {control.name!r} has incorrect region adjacency.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                    entity_ids=(int(face_ids[int(face_index)]),),
                )
            valid.append(int(face_index))
        face_indices = np.asarray(valid, dtype=np.int64)
        if not face_indices.size:
            if control.required:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Required patch {control.name!r} is absent.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            continue
        claimed.update(valid)
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            face_entity_set.entity_set_id,
            face_ids[face_indices],
        )
        patches.append(
            MeshPatch(
                control.name,
                scope,
                connected=_patch_is_connected(connectivity, face_indices),
                adjacent_zone_ids=tuple(
                    zone_by_name[name].zone_id for name in control.adjacent_region_names
                ),
            )
        )
    for face_index, adjacent in enumerate(incidents):
        if len(adjacent) != 2:
            continue
        regions = {
            str(region_by_cell[adjacent[0]]),
            str(region_by_cell[adjacent[1]]),
        }
        if len(regions) == 2 and face_index not in claimed:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated inter-region mesh face {face_index} is not declared by a PatchControl.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                entity_ids=(int(face_ids[face_index]),),
            )
    return tuple(zones), tuple(patches)


def _edge_size_evidence(
    control: UniformSizeControl,
    edges: np.ndarray,
    points: np.ndarray,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> tuple[list[str], tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(
        points[unique_edges[:, 1]] - points[unique_edges[:, 0]], axis=1
    )
    if not lengths.size:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A size control has no generated mesh edges.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    vertex_minimum = np.full((points.shape[0],), np.inf)
    vertex_maximum = np.zeros((points.shape[0],), dtype=np.float64)
    np.minimum.at(vertex_minimum, unique_edges[:, 0], lengths)
    np.minimum.at(vertex_minimum, unique_edges[:, 1], lengths)
    np.maximum.at(vertex_maximum, unique_edges[:, 0], lengths)
    np.maximum.at(vertex_maximum, unique_edges[:, 1], lengths)
    active = np.isfinite(vertex_minimum) & (vertex_minimum > 0.0)
    growth = float(np.max(vertex_maximum[active] / vertex_minimum[active], initial=1.0))
    local_minimum = float(np.min(lengths))
    local_maximum = float(np.max(lengths))
    key = f"size:{control.control_id}"
    requested = [(f"{key}:target_size", control.target_size)]
    achieved = [
        (f"{key}:minimum_edge", local_minimum),
        (f"{key}:maximum_edge", local_maximum),
        (f"{key}:maximum_local_edge_ratio", growth),
    ]
    for statistic in specification.size_compliance.target_statistics:
        quantile = {"p50": 0.5, "p95": 0.95}[statistic]
        achieved.append(
            (f"{key}:{statistic}_edge", float(np.quantile(lengths, quantile)))
        )
    optional = (
        ("minimum_size", control.minimum_size),
        ("maximum_size", control.maximum_size),
        ("maximum_growth_rate", control.maximum_growth_rate),
    )
    requested.extend(
        (f"{key}:{name}", value) for name, value in optional if value is not None
    )
    issues = []
    if control.strength is SizeControlStrength.HARD:
        policy = specification.size_compliance
        if control.minimum_size is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.minimum_size)
            )
            if local_minimum < control.minimum_size - tolerance:
                issues.append(f"minimum_size:{control.control_id}")
        if control.maximum_size is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.maximum_size)
            )
            if local_maximum > control.maximum_size + tolerance:
                issues.append(f"maximum_size:{control.control_id}")
        if control.maximum_growth_rate is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.maximum_growth_rate)
            )
            if growth > control.maximum_growth_rate + tolerance:
                issues.append(f"maximum_growth_rate:{control.control_id}")
    return issues, tuple(requested), tuple(achieved)


def _semantic_size_compliance(
    mesh: CellMesh,
    specification: VolumeMeshingSpec,
    cell_solid_ids: np.ndarray,
    /,
) -> tuple[list[str], tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    issues = []
    requested = []
    achieved = []
    for control in specification.size_controls:
        selected_edges = []
        cursor = 0
        solid_ids = np.asarray(control.scope.entity_ids, dtype=np.int64)
        for block in mesh.blocks:
            stop = cursor + block.cell_count
            selected = np.isin(cell_solid_ids[cursor:stop], solid_ids)
            if np.any(selected):
                cells = np.asarray(block.vertices, dtype=np.int32)[selected]
                pairs = np.asarray(
                    reference_cell_topology(block.cell_kind).entities[1],
                    dtype=np.int32,
                )
                selected_edges.append(cells[:, pairs].reshape((-1, 2)))
            cursor = stop
        if not selected_edges:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A solid-scoped size control has no generated cells.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        local_edges = np.concatenate(selected_edges)
        local_issues, local_requested, local_achieved = _edge_size_evidence(
            control, local_edges, points, specification
        )
        issues.extend(local_issues)
        requested.extend(local_requested)
        achieved.extend(local_achieved)
    return issues, tuple(requested), tuple(achieved)


def _execute_gmsh(gmsh, plan: GmshMeshingPlan, version: str, /) -> CellMeshingResult:
    source = plan.source
    specification = plan.specification
    options = plan.options
    report = source.report
    source_path = Path(report.source_id)
    if not source_path.is_file():
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Gmsh BRep meshing requires a reopenable STEP/IGES/BREP source path.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    shape, source_format, current_digest = read_occt_shape(source_path)
    if current_digest != report.source_digest or source_format != report.source_format:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The persisted BRep source bytes changed after import.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    limits = specification.limits
    if report.num_faces + report.num_edges + report.num_vertices > limits.maximum_faces:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "BRep entity count exceeds the meshing limit.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    minimum, target, maximum, curvature_points = _size_values(specification)
    dimension = specification.target.topological_dimension
    geometry_order = specification.target.geometry_order
    semantic_volume = isinstance(specification, VolumeMeshingSpec) and bool(
        specification.region_controls
    )
    semantic_surface = isinstance(specification, SurfaceMeshingSpec) and bool(
        specification.target.ambient_dimension == 2
        or specification.region_controls
        or specification.patch_controls
        or plan.planar_bands is not None
    )
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1 if options.terminal_output else 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.Algorithm", options.algorithm_2d)
    gmsh.option.setNumber("Mesh.Algorithm3D", options.algorithm_3d)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0 if minimum is None else minimum)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0e22 if maximum is None else maximum)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_points)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1 if target is not None else 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", geometry_order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    family_policy = specification.target.cell_families
    requested_kinds = {
        *family_policy.required,
        *family_policy.preferred,
        *family_policy.allowed_transitions,
    }
    pure_recombined = requested_kinds in ({"quadrilateral"}, {"hexahedron"})
    # Full-quad algorithm 3 halves every edge and cannot represent a single
    # exact normal layer. Blossom recombination preserves those transfinite
    # one-cell strips; later family compliance rejects unrecombined remainder.
    recombination_algorithm = (
        1 if pure_recombined and plan.planar_bands is not None else 3
    )
    gmsh.option.setNumber(
        "Mesh.RecombinationAlgorithm",
        recombination_algorithm if pure_recombined else 0,
    )
    gmsh.model.add(f"phydrax-{plan.plan_id[:12]}")
    gmsh.model.occ.importShapes(str(source_path))
    gmsh.model.occ.synchronize()
    if semantic_volume:
        cad_entities = _resolve_cad_entity_map(gmsh, source, shape)
    elif semantic_surface:
        cad_entities = _resolve_planar_cad_entity_map(gmsh, source, shape)
    else:
        cad_entities = None
    outside_size = 1.0e22 if target is None else target
    size_field_ids = _apply_uniform_size_fields(
        gmsh,
        source,
        shape,
        specification,
        cad_entities,
        outside_size,
    )
    sweep = (
        _prepare_swept_geometry(gmsh, plan, shape, cad_entities)
        if isinstance(specification, VolumeMeshingSpec)
        else None
    )
    if dimension == 2 and "quadrilateral" in requested_kinds:
        for entity_dimension, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(entity_dimension, tag)
    band_generation = _apply_planar_band_constraints(
        gmsh,
        plan.planar_bands,
        cad_entities,
        requested_kinds,
        target,
    )
    periodic_records = _set_periodic(gmsh, plan, shape)
    top_entities = sorted(gmsh.model.getEntities(dimension))
    if not top_entities:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "BRep source has no requested top-dimensional entities.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    if len(gmsh.model.getEntities()) > limits.maximum_faces:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Imported Gmsh entities exceed the declared limit.",
        )
    point_entities = gmsh.model.getEntities(0)
    if point_entities and target is not None:
        gmsh.model.mesh.setSize(point_entities, target)
    for entity_dimension, tag in top_entities:
        gmsh.model.addPhysicalGroup(entity_dimension, [tag], tag=tag)
    if sweep is None:
        gmsh.model.mesh.generate(dimension)
    else:
        gmsh.model.mesh.generate(2)
        _install_swept_cells(gmsh, sweep)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 0)

    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    order = np.argsort(node_tags, kind="stable")
    node_tags = node_tags[order]
    points = np.asarray(node_coordinates, dtype=np.float64).reshape((-1, 3))[order]
    if points.shape[0] > limits.maximum_vertices:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Generated Gmsh mesh exceeds maximum_vertices.",
        )
    top = _element_rows(gmsh, dimension, geometry_order)
    cell_count = sum(rows.tags.size for rows in top)
    if cell_count > limits.maximum_cells:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Generated Gmsh mesh exceeds maximum_cells.",
        )
    minimum_jacobian = _audit_jacobians(gmsh, top)
    periodic_requested, periodic_achieved = _audit_periodic(
        gmsh, periodic_records, node_tags, points
    )
    layer_audit = _audit_layers(sweep, top, node_tags, points)
    band_requested, band_achieved = _audit_planar_band_fronts(gmsh, band_generation)
    top_vertices = {
        rows.block_name: _local_connectivity(node_tags, rows.vertices) for rows in top
    }
    corner_nodes = np.unique(
        np.concatenate(
            [
                top_vertices[rows.block_name][:, : rows.corner_count].reshape(-1)
                for rows in top
            ]
        )
    )
    output_points = points
    if (
        isinstance(specification, SurfaceMeshingSpec)
        and specification.target.ambient_dimension == 2
    ):
        embedding = specification.planar_embedding
        if embedding is None:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SPECIFICATION,
                "Ambient-dimension-two execution lost its PlanarEmbedding.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        residuals = np.abs(embedding.plane_residual(points))
        plane_scale = max(
            1.0,
            float(np.max(np.abs(points), initial=0.0)),
            float(np.max(np.abs(np.asarray(embedding.origin)), initial=0.0)),
        )
        plane_tolerance = 8192.0 * np.finfo(np.float64).eps * plane_scale
        if np.max(residuals, initial=0.0) > plane_tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                "CAD-associated Gmsh nodes do not lie in the declared planar embedding.",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            )
        output_points = embedding.to_planar(points)
    corner_points = output_points[corner_nodes]
    corner_order = np.lexsort(
        tuple(
            corner_points[:, column]
            for column in range(corner_points.shape[1] - 1, -1, -1)
        )
    )
    corner_nodes = corner_nodes[corner_order]
    source_to_corner = np.full((points.shape[0],), -1, dtype=np.int32)
    source_to_corner[corner_nodes] = np.arange(corner_nodes.size, dtype=np.int32)
    mesh_points = output_points[corner_nodes]
    persistent_vertex_ids = np.arange(corner_nodes.size, dtype=np.int64)
    row_orders: dict[str, np.ndarray] = {}
    blocks = []
    next_cell_id = 0
    for rows in top:
        corners = source_to_corner[top_vertices[rows.block_name][:, : rows.corner_count]]
        keys = np.sort(corners, axis=1)
        row_order = np.lexsort(
            tuple(keys[:, column] for column in range(keys.shape[1] - 1, -1, -1))
        )
        row_orders[rows.block_name] = row_order
        blocks.append(
            CellBlock(
                rows.block_name,
                rows.cell_kind,
                corners[row_order],
                global_ids=np.arange(
                    next_cell_id,
                    next_cell_id + rows.tags.size,
                    dtype=np.int64,
                ),
            )
        )
        next_cell_id += rows.tags.size
    mesh = CellMesh(
        mesh_points,
        tuple(blocks),
        vertex_global_ids=persistent_vertex_ids,
        numeric_version=report.source_revision,
    )
    boundary_rows = _element_rows(gmsh, 2, geometry_order) if dimension == 3 else top
    mixed_boundary = any(rows.cell_kind == "quadrilateral" for rows in boundary_rows)
    boundary = None
    if not semantic_volume and specification.target.ambient_dimension == 3:
        boundary_triangles = []
        for rows in boundary_rows:
            corners = source_to_corner[
                _local_connectivity(node_tags, rows.vertices[:, : rows.corner_count])
            ]
            if np.any(corners < 0):
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    "Boundary corner is absent from the volume mesh.",
                )
            splits = (
                ((0, 1, 2),) if rows.cell_kind == "triangle" else ((0, 1, 2), (0, 2, 3))
            )
            for split in splits:
                boundary_triangles.append(corners[:, split])
        boundary_triangles = np.concatenate(boundary_triangles)
        boundary_keys = np.sort(boundary_triangles, axis=1)
        boundary_order = np.lexsort(
            tuple(
                boundary_keys[:, column]
                for column in range(boundary_keys.shape[1] - 1, -1, -1)
            )
        )
        boundary_tags = np.arange(boundary_triangles.shape[0], dtype=np.int64)
        boundary_metadata = SurfaceMetadata(
            source_id=report.source_id,
            source_revision=report.source_revision,
            coordinate_contract=source.coordinate_contract,
            provenance=("gmsh-occ", plan.plan_id),
            cell_tags=("gmsh-occ-surface",) * boundary_triangles.shape[0],
        )
        boundary = SurfaceModel.from_triangles(
            mesh_points,
            boundary_triangles[boundary_order],
            boundary_metadata,
            vertex_global_ids=persistent_vertex_ids,
            cell_global_ids=boundary_tags,
            numeric_version=report.source_revision,
            repair_orientation=True,
            orient_closed_outward=dimension == 3,
        )
        if dimension == 2 and not mixed_boundary and not semantic_surface:
            mesh = boundary.mesh
    mesh = canonicalize_cell_mesh(mesh)
    patches: tuple[MeshPatch, ...] = ()
    layer_attributes = _layer_attributes(mesh, top, row_orders, layer_audit)
    planar_evidence = None
    layer_interface_achieved: tuple[tuple[str, float], ...] = ()
    if semantic_volume:
        if cad_entities is None:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Semantic volume execution lost its resolved CAD entity map.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        cell_solid_ids = _canonical_cell_solid_ids(mesh, top, row_orders, cad_entities)
        surface_evidence = _semantic_surface_evidence(
            gmsh,
            source,
            mesh,
            boundary_rows,
            node_tags,
            source_to_corner,
            cell_solid_ids,
            cad_entities,
            plan.plan_id,
        )
        boundary = surface_evidence.boundary
        layer_interface_achieved = _audit_swept_interfaces(
            mesh,
            source,
            cell_solid_ids,
            surface_evidence.mesh_face_source,
            sweep,
        )
        region_zones, patches = _region_evidence(
            source,
            mesh,
            specification,
            cell_solid_ids,
            surface_evidence.mesh_face_source,
        )
        cell_entity_set = mesh.entity_set(3)
        cell_association = GeometryAssociation(
            GeometryAssociationKind.BREP,
            report.source_id,
            report.source_revision,
            cell_entity_set.entity_set_id,
            cell_entity_set.entity_ids,
            tuple(
                f"{report.source_revision}:solid:{int(owner)}" for owner in cell_solid_ids
            ),
            np.zeros((cell_solid_ids.size,), dtype=np.float64),
            exact=True,
        )
        zones = (*surface_evidence.zones, *region_zones)
        associations = (surface_evidence.association, cell_association)
        attributes = (surface_evidence.attribute, *layer_attributes)
    elif semantic_surface:
        if cad_entities is None:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Semantic planar execution lost its resolved CAD entity map.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        planar_evidence = _planar_surface_evidence(
            gmsh,
            source,
            mesh,
            specification,
            top,
            row_orders,
            node_tags,
            source_to_corner,
            cad_entities,
            geometry_order,
        )
        zones = planar_evidence.zones
        patches = planar_evidence.patches
        associations = planar_evidence.associations
        attributes = (*planar_evidence.attributes, *layer_attributes)
    else:
        if dimension == 3 and boundary is None:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Volume execution lost its generated boundary mesh.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        association, boundary_zones, provider_attribute = _boundary_association(
            source,
            mesh if dimension == 2 else boundary.mesh,
            options.association_tolerance_factor,
        )
        zones = boundary_zones if dimension == 2 else ()
        associations = (association,)
        attributes = (provider_attribute, *layer_attributes)
    if geometry_order == 1:
        geometry = CellGeometrySpec.affine(mesh)
    else:
        elements = {}
        routes = {}
        geometry_ordering = np.lexsort(
            tuple(
                output_points[:, column]
                for column in range(output_points.shape[1] - 1, -1, -1)
            )
        )
        point_to_geometry = np.empty((output_points.shape[0],), dtype=np.int32)
        point_to_geometry[geometry_ordering] = np.arange(
            output_points.shape[0], dtype=np.int32
        )
        for rows in top:
            element = lagrange_element(rows.cell_kind, geometry_order)
            route = top_vertices[rows.block_name][row_orders[rows.block_name]][
                :, _geometry_permutation(gmsh, rows, element)
            ]
            if dimension == 2 and not mixed_boundary:
                expected = corner_nodes[
                    np.asarray(mesh.block(rows.block_name).vertices, dtype=np.int32)
                ]
                flipped = np.any(route[:, :3] != expected, axis=1)
                if np.any(flipped):
                    reference = np.asarray(element.reference_nodes)
                    matches = (
                        np.max(
                            np.abs(reference[:, None] - reference[None, :, ::-1]), axis=-1
                        )
                        <= 2.0e-12
                    )
                    route[flipped] = route[flipped][:, np.argmax(matches, axis=1)]
            elements[rows.block_name] = element
            routes[rows.block_name] = point_to_geometry[route]
        geometry = CellGeometrySpec(elements, routes, output_points[geometry_ordering])
    audit, compliance = _audit_gmsh_mesh(
        specification,
        mesh,
        geometry,
        boundary,
        attributes,
        associations,
        zones,
        patches,
        region_zones,
        cell_solid_ids,
        family_policy,
        requested_kinds,
        minimum_jacobian,
        semantic_surface,
        semantic_volume,
        periodic_requested,
        periodic_achieved,
        layer_audit,
        layer_interface_achieved,
        band_requested,
        band_achieved,
        size_field_ids,
    )
    if not compliance.passed:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "; ".join(compliance.issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    semantic_control_ids = (
        *(control.control_id for control in specification.region_controls),
        *(control.control_id for control in specification.patch_controls),
    )
    band_control_ids = (
        ()
        if plan.planar_bands is None
        else tuple(control.control_id for control in plan.planar_bands.controls)
    )
    stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(report.source_revision,),
            output_ids=(plan.support.source_descriptor_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SCOPE_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=(specification.specification_id,),
            output_ids=(
                (
                    specification.scope.scope_id
                    if isinstance(specification, SurfaceMeshingSpec)
                    else specification.boundary_scope.scope_id
                ),
            ),
        ),
        MeshingStageReport(
            MeshingStageKind.CONTROL_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=(
                *(control.control_id for control in specification.size_controls),
                *semantic_control_ids,
                *band_control_ids,
            ),
            output_ids=(plan.plan_id,),
        ),
        *(
            (
                MeshingStageReport(
                    MeshingStageKind.LAYER_GENERATION,
                    MeshingStageStatus.PASSED,
                    input_ids=(
                        tuple(value.control_id for value in specification.layer_controls)
                        if sweep is not None
                        else band_control_ids
                    ),
                    output_ids=(mesh.mesh_id,),
                    created_count=cell_count,
                ),
            )
            if sweep is not None or band_generation is not None
            else ()
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING
            if dimension == 2
            else MeshingStageKind.VOLUME_FILL,
            MeshingStageStatus.PASSED,
            input_ids=(plan.plan_id,),
            output_ids=(mesh.mesh_id,),
            created_count=cell_count,
        ),
        MeshingStageReport(
            MeshingStageKind.CANONICALIZATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(mesh.topology_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.GEOMETRY_ASSOCIATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=tuple(value.association_id for value in associations),
        ),
        MeshingStageReport(
            MeshingStageKind.QUALITY_EVALUATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(audit.quality.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.GEOMETRY_AUDIT,
            MeshingStageStatus.PASSED,
            input_ids=(geometry.geometry_layout_id,),
            output_ids=(audit.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.TOPOLOGY_AUDIT,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.topology_id,),
            output_ids=(audit.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SPECIFICATION_COMPLIANCE,
            MeshingStageStatus.PASSED,
            input_ids=(specification.specification_id,),
            output_ids=(compliance.report_id,),
        ),
    )
    trace = MeshingTrace(stages)
    runtime = MeshingRuntimeInfo(
        plan.support.provider_id,
        version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=True,
        enforced_limits=("entities", "vertices", "cells"),
        unenforced_limits=("provider_workspace", "converted_arrays", "wall_time"),
    )
    provenance = SemanticProvenance(
        {
            "kind": "gmsh-cell-meshing-result",
            "source_revision": report.source_revision,
            "plan": plan.plan_id,
            "mesh": mesh.mesh_id,
            "associations": tuple(value.association_id for value in associations),
            "zones": tuple(value.zone_id for value in zones),
            "patches": tuple(value.patch_id for value in patches),
        },
        resource_ids={"source": report.source_id},
    )
    return CellMeshingResult(
        mesh,
        geometry,
        source.coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        GmshProvider(options).info,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        boundary=boundary,
        patches=patches,
        zones=zones,
        attributes=attributes,
        associations=associations,
    )


__all__ = [
    "GmshMeshingPlan",
    "GmshOptions",
    "GmshProvider",
    "GmshSession",
]
