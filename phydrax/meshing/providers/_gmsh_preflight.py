#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict Gmsh support preflight: every rejection is reported before generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...geometry.brep import BRepModel, BRepSource
from ...geometry.surface import SurfaceModel
from .._contracts import (
    MeshingSourceDescriptor,
    SurfaceMeshingSpec,
    SurfaceRemeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import (
    BackgroundMetricControl,
    BackgroundMetricMode,
    BoundaryLayerControl,
    BoundaryLayerCornerPolicy,
    BoundaryLayerRoute,
    FeatureKind,
    SurfaceReconstructionControl,
)
from .._organization import RegionRole
from .._planar_bands import PlanarBandResult
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    SizeCombinationPolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from ._gmsh_inventory import _cad_occurrence_inventory, _cad_scope_set
from ._gmsh_options import (
    GmshHighOrderOptimization,
    GmshSurfaceAlgorithm,
    GmshVolumeAlgorithm,
)


def _scope_indices(
    source: BRepModel, scope: MeshingScope, dimension: int, /
) -> np.ndarray | None:
    if dimension not in (0, 1, 2, 3):
        return None
    count = len(_cad_occurrence_inventory(source).entities[dimension])
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    if (
        scope.source_id != source.report.source_id
        or scope.source_revision != source.report.source_revision
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != dimension
        or scope.entity_set_id != _cad_scope_set(source, dimension)
        or np.any(identifiers < 0)
        or np.any(identifiers >= count)
    ):
        return None
    return identifiers


def _sweep_preflight_issues(
    source: BRepModel,
    controls: tuple[BoundaryLayerControl, ...],
    requested_kinds: set[str],
    geometry_order: int,
    /,
) -> list[str]:
    issues = []
    if geometry_order != 1:
        issues.append("Gmsh swept layers support affine order-one geometry only")
    controlled: dict[int, tuple[int, BoundaryLayerControl]] = {}
    face_roles: dict[tuple[int, int], str] = {}
    topology = _cad_occurrence_inventory(source).topology
    for control_index, control in enumerate(controls):
        # ty: ignore[invalid-argument-type]
        volume_ids = _scope_indices(source, control.volume_scope, 3)
        source_faces = _scope_indices(source, control.wall_scope, 2)
        # ty: ignore[invalid-argument-type]
        target_faces = _scope_indices(source, control.cap_scope, 2)
        if volume_ids is None or not volume_ids.size:
            issues.append("EXACT_SWEEP volume scope is not a nonempty solid scope")
            continue
        if source_faces is None or target_faces is None:
            issues.append("EXACT_SWEEP wall and cap are not face scopes of this BRep")
            continue
        volumes = {int(value) for value in volume_ids}
        if volumes & set(controlled):
            issues.append("EXACT_SWEEP volume scopes must be disjoint")
        for solid in volumes:
            controlled.setdefault(solid, (control_index, control))
        for role, faces in (("source", source_faces), ("target", target_faces)):
            assigned = set()
            for face_value in faces:
                face = int(face_value)
                owners = set(topology.face_solids[face])
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
    all_solids = set(range(topology.num_solids))
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
        for face in topology.solid_faces[solid]:
            role = face_roles.get((solid, face), "lateral")
            other_controlled = {
                owner for owner in topology.face_solids[face] if owner != solid
            } & selected_solids
            other_unswept = {
                owner for owner in topology.face_solids[face] if owner != solid
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


def _closed_wall(source: BRepModel, faces: np.ndarray, /) -> bool:
    """Every edge of the wall faces is shared only by wall faces."""
    wall = {int(face) for face in faces}
    topology = _cad_occurrence_inventory(source).topology
    return all(
        set(topology.edge_faces[edge]) <= wall
        for face in wall
        for edge in topology.face_edges[face]
    )


_LAYERED_VOLUME_KINDS = frozenset(("prism", "tetrahedron", "pyramid", "hexahedron"))


def _layered_volume_issues(
    source: BRepModel,
    specification: VolumeMeshingSpec,
    requested_kinds: set[str],
    geometry_order: int,
    /,
) -> list[str]:
    """ADVANCING and PROVIDER layers grow from closed walls of one solid."""
    controls = specification.layer_controls
    issues = []
    if len(controls) != 1:
        issues.append("Gmsh grows one advancing or provider boundary-layer control")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append("Advancing and provider layers fill their core with SIMPLEX cells")
    if len(_cad_occurrence_inventory(source).entities[3]) != 1:
        issues.append("Advancing and provider layers require one source solid")
    if (
        specification.region_controls
        or specification.patch_controls
        or specification.periodic_constraints
        or specification.protected_features
    ):
        issues.append(
            "Advancing and provider layers take no region, patch, periodic, or protected controls"
        )
    if geometry_order != 1:
        issues.append(
            "Advancing and provider layers support affine order-one geometry only"
        )
    if not {"prism", "tetrahedron"} <= requested_kinds <= _LAYERED_VOLUME_KINDS:
        issues.append(
            "Layered volumes require prisms and tetrahedra and allow only pyramid and hexahedron transitions"
        )
    if not specification.target.cell_families.allow_mixed:
        issues.append("Layered volumes require a mixed cell-family policy")
    if any(
        not isinstance(control, UniformSizeControl)
        or control.scope.scope_id != specification.boundary_scope.scope_id
        for control in specification.size_controls
    ):
        issues.append(
            "Layered volumes size their boundary surface with whole-source UniformSizeControl values only"
        )
    for control in controls:
        faces = _scope_indices(source, control.wall_scope, 2)
        if faces is None or not faces.size:
            issues.append("Boundary-layer wall scope is not a face scope of this BRep")
        elif not _closed_wall(source, faces):
            issues.append(
                "Advancing and provider walls must form closed wall components; open rims require EXACT_SWEEP"
            )
        if (
            control.route is BoundaryLayerRoute.PROVIDER
            and control.corner is BoundaryLayerCornerPolicy.FAN
        ):
            issues.append(
                "Gmsh boundary-layer extrusion has no fan templates; request SMOOTH or REJECT corners"
            )
    return issues


def _volume_layer_issues(
    source: BRepModel,
    specification: VolumeMeshingSpec,
    requested_kinds: set[str],
    geometry_order: int,
    /,
) -> list[str]:
    controls = specification.layer_controls
    routes = {control.route for control in controls}
    if len(routes) != 1:
        return ["Gmsh realizes one boundary-layer route per specification"]
    match routes.pop():
        case BoundaryLayerRoute.EXACT_SWEEP:
            issues = []
            if specification.fill_strategy is not VolumeFillStrategy.SWEEP:
                issues.append("EXACT_SWEEP layers require explicit straight SWEEP fill")
            if specification.periodic_constraints:
                issues.append("Swept layers cannot be combined with periodic constraints")
            issues.extend(
                _sweep_preflight_issues(source, controls, requested_kinds, geometry_order)
            )
            return issues
        case BoundaryLayerRoute.CAD_EXTRUSION:
            return [
                "CAD_EXTRUSION layers are partitioned by prepare_boundary_layer_extrusion and meshed as EXACT_SWEEP"
            ]
        case BoundaryLayerRoute.ADVANCING | BoundaryLayerRoute.PROVIDER:
            return _layered_volume_issues(
                source, specification, requested_kinds, geometry_order
            )
        case route:
            raise TypeError(f"Unsupported boundary-layer route {route!r}.")


def _surface_layer_issues(
    source: BRepModel,
    specification: SurfaceMeshingSpec,
    requested_kinds: set[str],
    planar_bands: bool,
    /,
) -> list[str]:
    """Planar provider layers lower to one Gmsh BoundaryLayer field."""
    controls = specification.layer_controls
    if not controls:
        return []
    issues = []
    if len(controls) != 1:
        issues.append("Gmsh lowers one planar boundary-layer field per specification")
    if specification.target.ambient_dimension != 2 or planar_bands:
        issues.append(
            "Planar boundary layers require an ambient-dimension-two BRep without planar bands"
        )
    if specification.target.geometry_order != 1:
        issues.append("Planar boundary layers support affine order-one geometry only")
    if specification.periodic_constraints:
        issues.append(
            "Planar boundary layers cannot be combined with periodic constraints"
        )
    if not requested_kinds <= {"triangle", "quadrilateral"}:
        issues.append("Planar boundary layers produce triangles and quadrilaterals")
    for control in controls:
        if _scope_indices(source, control.wall_scope, 1) is None:
            issues.append("Planar boundary-layer walls must be edge scopes of this BRep")
        rates = np.asarray(control.schedule.growth_rates, dtype=np.float64)
        if rates.size and np.ptp(rates) > 1.0e-12 * np.max(rates):
            issues.append("Gmsh boundary-layer fields realize geometric schedules only")
    return issues


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
    inventory = _cad_occurrence_inventory(source)
    topology = inventory.topology
    face_regions = np.empty((len(inventory.entities[2]),), dtype=object)
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
    expected_faces = set(range(len(inventory.entities[2])))
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
            owners = topology.edge_faces[edge]
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
        for edge, owners in enumerate(topology.edge_faces)
        if len(owners) == 1
        or (len(owners) == 2 and face_regions[owners[0]] != face_regions[owners[1]])
    }
    missing = expected_patches - declared
    if missing:
        issues.append(
            f"Source BRep contains undeclared exterior/interface edges {sorted(missing)}"
        )
    return issues


def _semantic_region_issues(
    specification: VolumeMeshingSpec, model: BRepModel, unsupported: list[str], /
) -> None:
    if not specification.region_controls:
        unsupported.append("Every solid requires an explicit RegionControl")
    occupied: set[int] = set()
    region_names: set[str] = set()
    solid_region: dict[int, str] = {}
    topology = _cad_occurrence_inventory(model).topology
    for control in specification.region_controls:
        identifiers = _scope_indices(model, control.scope, 3)
        if identifiers is None:
            unsupported.append("RegionControl scope is not a solid scope of this BRep")
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
            unsupported.append("Disabled and void RegionControl values are unsupported")
    if occupied != set(range(topology.num_solids)):
        unsupported.append(
            "RegionControl scopes must exhaustively cover all source solids"
        )
    if occupied == set(range(topology.num_solids)):
        observed_internal_faces = {
            face
            for face, owners in enumerate(topology.face_solids)
            if len(owners) == 2 and solid_region[owners[0]] != solid_region[owners[1]]
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
                owners = topology.face_solids[int(face)]
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


def _validate_gmsh_semantics(
    source: Any,
    specification: Any,
    model: Any,
    descriptor: MeshingSourceDescriptor,
    target: Any,
    requested: set[str],
    layers: Any,
    unsupported: list[str],
    /,
) -> bool:
    semantic_volume = False
    topology = _cad_occurrence_inventory(model).topology
    if isinstance(specification, VolumeMeshingSpec):
        semantic_volume = bool(
            topology.num_solids > 1
            or specification.region_controls
            or specification.patch_controls
        )
        if not descriptor.closed:
            unsupported.append(
                "Gmsh volume meshing requires at least one closed BRep solid"
            )
        # Layered volumes resolve their own boundary; they need only one closed solid.
        layered = any(
            control.route in (BoundaryLayerRoute.ADVANCING, BoundaryLayerRoute.PROVIDER)
            for control in layers
        )
        if (
            isinstance(source, BRepModel)
            and not isinstance(source, BRepSource)
            and not semantic_volume
            and not layered
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
            unsupported.extend(
                _volume_layer_issues(
                    model, specification, requested, target.geometry_order
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
            _semantic_region_issues(specification, model, unsupported)
    else:
        if specification.quality_target is not None:
            unsupported.append("Gmsh does not enforce a MeshQualityTarget")
        semantic_surface = bool(
            target.ambient_dimension == 2
            or specification.region_controls
            or specification.patch_controls
            or isinstance(source, PlanarBandResult)
        )
        if semantic_surface:
            unsupported.extend(_semantic_surface_issues(model, specification))
        elif topology.num_solids > 1:
            unsupported.append(
                "Multi-solid BRep surface meshing is outside the strict semantic volume path"
            )
        unsupported.extend(
            _surface_layer_issues(
                model,
                specification,
                requested,
                isinstance(source, PlanarBandResult),
            )
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


def _validate_gmsh_periodicity(
    specification: Any, model: Any, unsupported: list[str], /
) -> None:
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


def _proximity_issues(
    model: BRepModel, specification: Any, control: ProximitySizeControl, /
) -> list[str]:
    dimension = control.source_scope.entity_dimension
    if dimension not in (1, 2):
        return ["Gmsh proximity sizing measures gaps between source curves or faces"]
    source_ids = _scope_indices(model, control.source_scope, dimension)
    target_ids = _scope_indices(model, control.target_scope, dimension)
    if source_ids is None or target_ids is None:
        return ["ProximitySizeControl scopes are not entity scopes of this BRep"]
    issues = []
    if np.intersect1d(source_ids, target_ids).size:
        issues.append("ProximitySizeControl source and target scopes must be disjoint")
    if specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY:
        issues.append(
            "Gmsh lowers proximity sizing through minimum combination, not explicit priority"
        )
    return issues


def _validate_gmsh_size_controls(
    specification: Any,
    model: Any,
    scope: Any,
    semantic_volume: bool,
    unsupported: list[str],
    /,
) -> None:
    covered_solids: set[int] = set()
    for control in specification.size_controls:
        if isinstance(control, ProximitySizeControl):
            unsupported.extend(_proximity_issues(model, specification, control))
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
                    "Semantic volume sizing supports solid-scoped uniform and proximity controls only"
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
    if semantic_volume and covered_solids != set(
        range(len(_cad_occurrence_inventory(model).entities[3]))
    ):
        unsupported.append(
            "Solid-scoped UniformSizeControl values must cover every source solid"
        )


_FEATURE_DIMENSIONS = {
    FeatureKind.CORNER: 0,
    FeatureKind.CURVE: 1,
    FeatureKind.SURFACE: 2,
    FeatureKind.MATERIAL_INTERFACE: 2,
}


def _protected_feature_issues(
    model: BRepModel,
    specification: Any,
    semantic: bool,
    /,
) -> list[str]:
    """Protected corners and curves are CAD-conforming or embedded; faces are bounded."""
    issues = []
    volume = isinstance(specification, VolumeMeshingSpec)
    constrained = bool(
        (volume and specification.layer_controls) or specification.periodic_constraints
    )
    topology = _cad_occurrence_inventory(model).topology
    for feature in specification.protected_features:
        dimension = _FEATURE_DIMENSIONS.get(feature.feature_kind)
        if dimension is None:
            issues.append("Gmsh protected feature kind is unsupported")
            continue
        identifiers = (
            _scope_indices(model, feature.scope, dimension)
            if feature.scope.entity_dimension == dimension
            else None
        )
        if identifiers is None:
            issues.append(
                f"{feature.feature_kind.value} protected features require a dimension-{dimension} scope of this BRep"
            )
            continue
        if dimension == 2 and not volume:
            issues.append(
                "Surface meshing conforms to every source face; protect curves or corners"
            )
            continue
        if dimension == 2:
            owners = tuple(topology.face_solids[int(face)] for face in identifiers)
            if any(not values for values in owners):
                issues.append("Gmsh embeds free protected curves and points only")
            if feature.feature_kind is FeatureKind.MATERIAL_INTERFACE and any(
                len(values) != 2 for values in owners
            ):
                issues.append(
                    "Material-interface protected features must select faces shared by two solids"
                )
            continue
        if dimension == 1 and any(
            not topology.edge_faces[int(edge)] for edge in identifiers
        ):
            if semantic:
                issues.append(
                    "Strict semantic Gmsh paths do not embed free protected curves"
                )
            if constrained:
                issues.append(
                    "Embedded protected curves cannot be combined with sweeps or periodicity"
                )
    return issues


def _background_metric_issues(
    options: Any,
    model: BRepModel,
    specification: Any,
    background: BackgroundMetricControl,
    local_sizing: bool,
    /,
) -> list[str]:
    issues = []
    target = specification.target
    if background.coordinate_contract.spatial_id != model.coordinate_contract.spatial_id:
        issues.append(
            "BackgroundMetricControl coordinates differ from the source coordinate contract"
        )
    if background.mesh.ambient_dimension != 3 or target.ambient_dimension != 3:
        issues.append(
            "Gmsh background metrics require three-dimensional source coordinates"
        )
        return issues
    source_points = np.asarray(model.mesh_vertices, dtype=np.float64)
    background_points = np.asarray(background.mesh.coordinates, dtype=np.float64)
    tolerance = model.report.linear_deflection * options.association_tolerance_factor
    if np.any(
        np.min(source_points, axis=0) < np.min(background_points, axis=0) - tolerance
    ) or np.any(
        np.max(source_points, axis=0) > np.max(background_points, axis=0) + tolerance
    ):
        issues.append("BackgroundMetricControl mesh does not cover the source bounds")
    match background.mode:
        case BackgroundMetricMode.ISOTROPIC:
            return issues
        case BackgroundMetricMode.ANISOTROPIC:
            pass
        case _:
            raise ValueError("Unsupported background metric mode.")
    if isinstance(specification, VolumeMeshingSpec):
        issues.append(
            "Gmsh does not lower anisotropic volume metrics; use MmgProvider adaptation"
        )
        return issues
    if options.algorithm_2d is not GmshSurfaceAlgorithm.BAMG:
        issues.append(
            "Anisotropic Gmsh surface metrics require the BAMG surface algorithm"
        )
    requested = {
        *target.cell_families.required,
        *target.cell_families.preferred,
        *target.cell_families.allowed_transitions,
    }
    if requested != {"triangle"}:
        issues.append("Anisotropic BAMG surface metrics generate triangles only")
    if (
        local_sizing
        or specification.periodic_constraints
        or specification.protected_features
    ):
        issues.append(
            "Anisotropic metrics are the sole Gmsh size field; local, proximity, curvature, periodic, and protected controls are unsupported"
        )
    if any(
        isinstance(control, UniformSizeControl)
        and control.target_size < background.metric.maximum_size
        for control in specification.size_controls
    ):
        issues.append(
            "Uniform targets below the metric's maximum size would override anisotropic sizing"
        )
    return issues


def _generation_option_issues(options: Any, specification: Any, /) -> list[str]:
    issues = []
    volume = isinstance(specification, VolumeMeshingSpec)
    layers = volume and bool(specification.layer_controls)
    if options.num_threads > 1 and specification.deterministic:
        issues.append(
            "Deterministic specifications require single-threaded Gmsh generation"
        )
    if options.algorithm_3d is GmshVolumeAlgorithm.HXT and layers:
        issues.append("HXT does not fill swept-layer remainders")
    if options.optimize_netgen and (not volume or layers):
        issues.append("Netgen optimization applies to unswept tetrahedral volume meshes")
    if (
        options.high_order_optimization is not GmshHighOrderOptimization.NONE
        and specification.target.geometry_order == 1
    ):
        issues.append("High-order optimization requires geometry order at least two")
    return issues


def _brep_support_issues(
    options: Any,
    source: Any,
    model: BRepModel,
    descriptor: MeshingSourceDescriptor,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    background: BackgroundMetricControl | None,
    /,
) -> list[str]:
    unsupported = []
    target = specification.target
    scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
    dimension = target.topological_dimension
    inventory = _cad_occurrence_inventory(model)
    scope_ids = _scope_indices(model, scope, dimension)
    expected_ids = np.arange(
        len(inventory.entities[dimension]),
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
        if target.ambient_dimension == 2 and inventory.entities[3]:
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
        else _LAYERED_VOLUME_KINDS
        if any(
            control.route in (BoundaryLayerRoute.ADVANCING, BoundaryLayerRoute.PROVIDER)
            for control in layers
        )
        else {"prism", "tetrahedron"}
        if layers
        else {"tetrahedron"}
    )
    if not requested <= supported:
        unsupported.append(f"Gmsh selected path supports only {sorted(supported)}")
    if target.geometry_order not in (1, 2, 3, 4):
        unsupported.append(
            "Gmsh complete geometry elements support orders one through four"
        )
    elif target.geometry_order > 2 and not requested <= {"triangle", "tetrahedron"}:
        unsupported.append(
            "Gmsh geometry orders three and four support simplex families only"
        )
    semantic_volume = _validate_gmsh_semantics(
        source,
        specification,
        model,
        descriptor,
        target,
        requested,
        layers,
        unsupported,
    )
    semantic = semantic_volume or (
        isinstance(specification, SurfaceMeshingSpec)
        and bool(
            target.ambient_dimension == 2
            or specification.region_controls
            or specification.patch_controls
            or isinstance(source, PlanarBandResult)
        )
    )
    unsupported.extend(_protected_feature_issues(model, specification, semantic))
    _validate_gmsh_periodicity(specification, model, unsupported)
    _validate_gmsh_size_controls(
        specification, model, scope, semantic_volume, unsupported
    )
    unsupported.extend(_generation_option_issues(options, specification))
    if background is not None:
        local_sizing = any(
            isinstance(control, (ProximitySizeControl, CurvatureSizeControl))
            or control.scope.scope_id != scope.scope_id
            for control in specification.size_controls
        )
        unsupported.extend(
            _background_metric_issues(
                options, model, specification, background, local_sizing
            )
        )
    return unsupported


def _remeshing_support_issues(
    options: Any,
    source: SurfaceModel,
    specification: SurfaceRemeshingSpec,
    background: BackgroundMetricControl | None,
    reconstruction: SurfaceReconstructionControl | None,
    /,
) -> list[str]:
    unsupported = []
    surface = specification.surface
    mesh = source.mesh
    target = surface.target
    scope = surface.scope
    if reconstruction is None:
        unsupported.append(
            "Discrete surface remeshing requires a SurfaceReconstructionControl"
        )
    if specification.source_mesh_id != mesh.mesh_id:
        unsupported.append("SurfaceRemeshingSpec does not bind the supplied source mesh")
    cell_set = mesh.entity_set(2)
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != 2
        or scope.entity_set_id != cell_set.entity_set_id
        or not np.array_equal(
            np.asarray(scope.entity_ids), np.sort(np.asarray(cell_set.entity_ids))
        )
    ):
        unsupported.append(
            "Gmsh remeshes the complete source surface; the scope must bind every source cell"
        )
    requested = {
        *target.cell_families.required,
        *target.cell_families.preferred,
        *target.cell_families.allowed_transitions,
    }
    if requested != {"triangle"} or target.ambient_dimension != 3:
        unsupported.append(
            "Gmsh discrete remeshing produces triangles in three dimensions"
        )
    if target.geometry_order not in (1, 2, 3, 4):
        unsupported.append(
            "Gmsh complete geometry elements support orders one through four"
        )
    if (
        surface.protected_features
        or surface.region_controls
        or surface.patch_controls
        or surface.periodic_constraints
        or surface.layer_controls
    ):
        unsupported.append(
            "Discrete remeshing preserves classified features only; semantic, protected, periodic, and layer controls are unsupported"
        )
    if any(
        isinstance(control, ProximitySizeControl)
        or control.scope.scope_id != scope.scope_id
        or (isinstance(control, CurvatureSizeControl) and control.use_faceted_curvature)
        for control in surface.size_controls
    ):
        unsupported.append(
            "Discrete remeshing supports whole-surface uniform and parametric curvature sizing only"
        )
    if (
        surface.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY
        and len(surface.size_controls) > 1
    ):
        unsupported.append(
            "Gmsh does not lower explicit priority across whole-surface controls"
        )
    unsupported.extend(_generation_option_issues(options, surface))
    if background is not None:
        if (
            background.coordinate_contract.spatial_id
            != source.metadata.coordinate_contract.spatial_id
        ):
            unsupported.append(
                "BackgroundMetricControl coordinates differ from the source coordinate contract"
            )
        if background.mode is not BackgroundMetricMode.ISOTROPIC:
            unsupported.append(
                "Discrete remeshing lowers isotropic background metrics only"
            )
    return unsupported
