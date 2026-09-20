#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry.brep import PlanarEmbedding
from ._controls import (
    HoleSeed,
    PatchControl,
    PeriodicConstraint,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
    SweptLayerControl,
)
from ._scope import MeshingScope
from ._sizing import (
    ProximitySizeControl,
    SizeCombinationPolicy,
    SizeCompliancePolicy,
    SizeControl,
)


class MeshingOperation(StrEnum):
    FACET_GEOMETRY = "facet_geometry"
    MESH_SURFACE = "mesh_surface"
    REMESH_SURFACE = "remesh_surface"
    REMESH_SURFACE_LOCALLY = "remesh_surface_locally"
    MESH_VOLUME = "mesh_volume"
    ADAPT_VOLUME = "adapt_volume"
    GENERATE_LAYERS = "generate_layers"
    SWEEP_VOLUME = "sweep_volume"
    WRAP_SURFACE = "wrap_surface"
    REPAIR_MESH = "repair_mesh"
    OPTIMIZE_MESH = "optimize_mesh"
    PARTITION_MESH = "partition_mesh"
    ASSEMBLE_OVERSET = "assemble_overset"
    BOOLEAN_SURFACE = "boolean_surface"


class MeshingSourceKind(StrEnum):
    BREP = "brep"
    IMPLICIT = "implicit"
    SURFACE = "surface"
    CELL_MESH = "cell_mesh"
    POINT_CLOUD = "point_cloud"
    IMAGE = "image"
    TENSOR_GRID = "tensor_grid"
    MESH_ASSEMBLY = "mesh_assembly"


class MeshingCapability(StrEnum):
    DETERMINISTIC = "deterministic"
    CAD_CONFORMING = "cad_conforming"
    IMPLICIT_CONFORMING = "implicit_conforming"
    SURFACE_CONSTRAINED = "surface_constrained"
    MULTI_MATERIAL = "multi_material"
    ANISOTROPIC_METRIC = "anisotropic_metric"
    BOUNDARY_LAYERS = "boundary_layers"
    PERIODIC = "periodic"
    HIGH_ORDER_GEOMETRY = "high_order_geometry"
    MIXED_CELLS = "mixed_cells"
    POLYHEDRAL = "polyhedral"
    LINEAGE = "lineage"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


class MeshingDerivativeMode(StrEnum):
    FIXED_TOPOLOGY_EXACT = "fixed_topology_exact"
    FIXED_ROUTE_PIECEWISE = "fixed_route_piecewise"
    FROZEN_EVENT_SCHEDULE = "frozen_event_schedule"
    CUSTOM_TRANSFER_PULLBACK = "custom_transfer_pullback"
    RELAXED_SURROGATE = "relaxed_surrogate"
    NONDIFFERENTIABLE = "nondifferentiable"


class MeshingExecutionMode(StrEnum):
    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    REMOTE = "remote"


class VolumeFillStrategy(StrEnum):
    SIMPLEX = "simplex"
    POLYHEDRAL = "polyhedral"
    HEX_DOMINANT_SIMPLEX_TRANSITION = "hex_dominant_simplex_transition"
    HEX_DOMINANT_POLYHEDRAL_TRANSITION = "hex_dominant_polyhedral_transition"
    SWEEP = "sweep"
    MULTIZONE = "multizone"


class MeshingFailureCategory(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_SPECIFICATION = "invalid_specification"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSUPPORTED_COMBINATION = "unsupported_combination"
    INVALID_SOURCE = "invalid_source"
    SCOPE_RESOLUTION_FAILED = "scope_resolution_failed"
    CONTROL_CONFLICT = "control_conflict"
    REGION_RESOLUTION_FAILED = "region_resolution_failed"
    PROVIDER_EXECUTION_FAILED = "provider_execution_failed"
    INTERRUPTED = "interrupted"
    TIMED_OUT = "timed_out"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CONVERSION_FAILED = "conversion_failed"
    CANONICALIZATION_FAILED = "canonicalization_failed"
    ASSOCIATION_FAILED = "association_failed"
    AUDIT_FAILED = "audit_failed"
    QUALITY_REJECTED = "quality_rejected"
    COMPLIANCE_FAILED = "compliance_failed"
    LINEAGE_FAILED = "lineage_failed"
    TRANSFER_FAILED = "transfer_failed"


class MeshingLimits(StrictModule, NonTrainableState):
    maximum_vertices: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_faces: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_connectivity_entries: int = eqx.field(static=True)
    maximum_data_bytes: int = eqx.field(static=True)
    maximum_wall_seconds: float = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_vertices: int = 10_000_000,
        maximum_edges: int = 50_000_000,
        maximum_faces: int = 50_000_000,
        maximum_cells: int = 20_000_000,
        maximum_connectivity_entries: int = 500_000_000,
        maximum_data_bytes: int = 4_000_000_000,
        maximum_wall_seconds: float = 3600.0,
    ):
        counts = (
            int(maximum_vertices),
            int(maximum_edges),
            int(maximum_faces),
            int(maximum_cells),
            int(maximum_connectivity_entries),
            int(maximum_data_bytes),
        )
        seconds = float(maximum_wall_seconds)
        if any(value <= 0 for value in counts):
            raise ValueError("Meshing entity and data limits must be positive.")
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("maximum_wall_seconds must be positive and finite.")
        (
            self.maximum_vertices,
            self.maximum_edges,
            self.maximum_faces,
            self.maximum_cells,
            self.maximum_connectivity_entries,
            self.maximum_data_bytes,
        ) = counts
        self.maximum_wall_seconds = seconds
        self.limits_id = canonical_fingerprint(
            {
                "kind": "meshing-limits",
                "counts": counts,
                "maximum_wall_seconds": seconds,
            }
        )


class CellFamilyPolicy(StrictModule, NonTrainableState):
    required: tuple[str, ...] = eqx.field(static=True)
    preferred: tuple[str, ...] = eqx.field(static=True)
    allowed_transitions: tuple[str, ...] = eqx.field(static=True)
    allow_mixed: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        required: tuple[str, ...] = (),
        preferred: tuple[str, ...] = (),
        allowed_transitions: tuple[str, ...] = (),
        allow_mixed: bool = False,
    ):
        supported = {
            "interval",
            "triangle",
            "quadrilateral",
            "polygon",
            "tetrahedron",
            "hexahedron",
            "prism",
            "pyramid",
            "polyhedron",
        }
        required_ = tuple(str(value) for value in required)
        preferred_ = tuple(str(value) for value in preferred)
        transitions = tuple(str(value) for value in allowed_transitions)
        values = (*required_, *preferred_, *transitions)
        if not required_ and not preferred_:
            raise ValueError(
                "At least one required or preferred cell family is required."
            )
        if any(value not in supported for value in values):
            raise ValueError("Cell family policy contains an unsupported canonical kind.")
        if len(set(required_)) != len(required_) or len(set(preferred_)) != len(
            preferred_
        ):
            raise ValueError("Cell family entries must be unique within each role.")
        if not allow_mixed and len(set(values)) > 1:
            raise ValueError("Multiple cell families require allow_mixed=True.")
        self.required = required_
        self.preferred = preferred_
        self.allowed_transitions = transitions
        self.allow_mixed = bool(allow_mixed)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cell-family-policy",
                "required": required_,
                "preferred": preferred_,
                "allowed_transitions": transitions,
                "allow_mixed": bool(allow_mixed),
            }
        )


class CellMeshingTarget(StrictModule, NonTrainableState):
    topological_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    cell_families: CellFamilyPolicy
    geometry_order: int = eqx.field(static=True)
    require_conforming: bool = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        topological_dimension: int,
        ambient_dimension: int,
        cell_families: CellFamilyPolicy,
        /,
        *,
        geometry_order: int = 1,
        require_conforming: bool = True,
    ):
        topological = int(topological_dimension)
        ambient = int(ambient_dimension)
        order = int(geometry_order)
        if topological <= 0 or ambient < topological:
            raise ValueError(
                "Meshing dimensions must satisfy 0 < topological <= ambient."
            )
        if not isinstance(cell_families, CellFamilyPolicy):
            raise TypeError("cell_families must be CellFamilyPolicy.")
        if order <= 0:
            raise ValueError("geometry_order must be positive.")
        self.topological_dimension = topological
        self.ambient_dimension = ambient
        self.cell_families = cell_families
        self.geometry_order = order
        self.require_conforming = bool(require_conforming)
        self.target_id = canonical_fingerprint(
            {
                "kind": "cell-meshing-target",
                "dimensions": [topological, ambient],
                "cell_families": cell_families.policy_id,
                "geometry_order": order,
                "require_conforming": bool(require_conforming),
            }
        )


def _size_control_scopes(control: SizeControl, /) -> tuple[MeshingScope, ...]:
    if isinstance(control, ProximitySizeControl):
        return control.source_scope, control.target_scope
    return (control.scope,)


def _validated_semantic_controls(
    target: CellMeshingTarget,
    scope: MeshingScope,
    size_controls,
    protected_features,
    region_controls,
    patch_controls,
    periodic_constraints,
    size_combination,
    size_compliance,
    /,
):
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    expected_scope_dimension = (
        target.topological_dimension - 1
        if isinstance(scope, MeshingScope)
        and target.topological_dimension == 3
        and scope.entity_dimension == 2
        else target.topological_dimension
    )
    if scope.entity_dimension != expected_scope_dimension:
        raise ValueError(
            "Top-level scope dimension is incompatible with the meshing target."
        )
    sizes = tuple(size_controls)
    features = tuple(protected_features)
    regions = tuple(region_controls)
    patches = tuple(patch_controls)
    periodic = tuple(periodic_constraints)
    if not sizes:
        raise ValueError("Meshing requires at least one size control.")
    if not all(isinstance(control, SizeControl) for control in sizes):
        raise TypeError("size_controls must contain SizeControl values.")
    if not all(isinstance(feature, ProtectedFeature) for feature in features):
        raise TypeError("protected_features must contain ProtectedFeature values.")
    if not all(isinstance(control, RegionControl) for control in regions):
        raise TypeError("region_controls must contain RegionControl values.")
    if not all(isinstance(control, PatchControl) for control in patches):
        raise TypeError("patch_controls must contain PatchControl values.")
    if not all(isinstance(constraint, PeriodicConstraint) for constraint in periodic):
        raise TypeError("periodic_constraints must contain PeriodicConstraint values.")
    if not isinstance(size_combination, SizeCombinationPolicy):
        raise TypeError("size_combination must be SizeCombinationPolicy.")
    compliance = SizeCompliancePolicy() if size_compliance is None else size_compliance
    if not isinstance(compliance, SizeCompliancePolicy):
        raise TypeError("size_compliance must be SizeCompliancePolicy or None.")

    binding = (scope.source_id, scope.source_revision, scope.entity_kind)
    scoped = (
        *(
            size_scope
            for control in sizes
            for size_scope in _size_control_scopes(control)
        ),
        *(feature.scope for feature in features),
        *(control.scope for control in regions),
        *(control.scope for control in patches),
        *(constraint.source_scope for constraint in periodic),
        *(constraint.target_scope for constraint in periodic),
    )
    if any(
        (value.source_id, value.source_revision, value.entity_kind) != binding
        for value in scoped
    ):
        raise ValueError("Meshing controls must share the top-level source binding.")
    dimension = target.topological_dimension
    if regions and scope.entity_dimension != dimension:
        raise ValueError("Region controls require a top-dimensional source scope.")
    if any(
        control.scope.entity_dimension != dimension
        or control.scope.entity_set_id != scope.entity_set_id
        or np.setdiff1d(control.scope.entity_ids, scope.entity_ids).size
        for control in regions
    ):
        raise ValueError(
            "Region controls must select contained top-dimensional entities."
        )
    if any(control.scope.entity_dimension != dimension - 1 for control in patches):
        raise ValueError("Patch controls must select codimension-one entities.")
    region_names = tuple(control.region_name for control in regions)
    if len(set(region_names)) != len(region_names):
        raise ValueError("Region control names must be unique.")
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            if np.intersect1d(first.scope.entity_ids, second.scope.entity_ids).size:
                raise ValueError("Region control scopes must be disjoint.")
    patch_names = tuple(control.name for control in patches)
    if len(set(patch_names)) != len(patch_names):
        raise ValueError("Patch control names must be unique.")
    for index, first in enumerate(patches):
        for second in patches[index + 1 :]:
            if (
                first.scope.entity_set_id == second.scope.entity_set_id
                and np.intersect1d(first.scope.entity_ids, second.scope.entity_ids).size
            ):
                raise ValueError("Patch control scopes must be disjoint.")
    declared = set(region_names)
    if any(not set(control.adjacent_region_names) <= declared for control in patches):
        raise ValueError("Patch controls must reference declared region names.")
    return sizes, features, regions, patches, periodic, compliance


class SurfaceMeshingSpec(StrictModule, NonTrainableState):
    target: CellMeshingTarget
    planar_embedding: PlanarEmbedding | None = eqx.field(static=True)
    scope: MeshingScope
    size_controls: tuple[SizeControl, ...]
    protected_features: tuple[ProtectedFeature, ...]
    region_controls: tuple[RegionControl, ...]
    patch_controls: tuple[PatchControl, ...]
    periodic_constraints: tuple[PeriodicConstraint, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: SizeCompliancePolicy
    limits: MeshingLimits
    deterministic: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: CellMeshingTarget,
        scope: MeshingScope,
        /,
        *,
        planar_embedding: PlanarEmbedding | None = None,
        size_controls: tuple[SizeControl, ...],
        protected_features: tuple[ProtectedFeature, ...] = (),
        region_controls: tuple[RegionControl, ...] = (),
        patch_controls: tuple[PatchControl, ...] = (),
        periodic_constraints: tuple[PeriodicConstraint, ...] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: SizeCompliancePolicy | None = None,
        limits: MeshingLimits | None = None,
        deterministic: bool = True,
    ):
        if not isinstance(target, CellMeshingTarget) or target.topological_dimension != 2:
            raise ValueError(
                "Surface meshing target must have topological dimension two."
            )
        if planar_embedding is not None and not isinstance(
            planar_embedding, PlanarEmbedding
        ):
            raise TypeError("planar_embedding must be a PlanarEmbedding or None.")
        if (target.ambient_dimension == 2) != (planar_embedding is not None):
            raise ValueError(
                "A PlanarEmbedding is required exactly for ambient-dimension-two surface targets."
            )
        (
            sizes,
            features,
            regions,
            patches,
            periodic,
            compliance,
        ) = _validated_semantic_controls(
            target,
            scope,
            size_controls,
            protected_features,
            region_controls,
            patch_controls,
            periodic_constraints,
            size_combination,
            size_compliance,
        )
        limit = MeshingLimits() if limits is None else limits
        if not isinstance(limit, MeshingLimits):
            raise TypeError("limits must be MeshingLimits or None.")
        self.target = target
        self.planar_embedding = planar_embedding
        self.scope = scope
        self.size_controls = sizes
        self.protected_features = features
        self.region_controls = regions
        self.patch_controls = patches
        self.periodic_constraints = periodic
        self.size_combination = size_combination
        self.size_compliance = compliance
        self.limits = limit
        self.deterministic = bool(deterministic)
        self.specification_id = canonical_fingerprint(
            {
                "kind": "surface-meshing-spec",
                "target": target.target_id,
                "planar_embedding": (
                    None if planar_embedding is None else planar_embedding.embedding_id
                ),
                "scope": scope.scope_id,
                "size_controls": [control.control_id for control in sizes],
                "size_combination": size_combination.value,
                "size_compliance": compliance.policy_id,
                "protected_features": [value.feature_id for value in features],
                "regions": [value.control_id for value in regions],
                "patches": [value.control_id for value in patches],
                "periodic": [value.constraint_id for value in periodic],
                "limits": limit.limits_id,
                "deterministic": bool(deterministic),
            }
        )


class SurfaceRemeshingSpec(StrictModule, NonTrainableState):
    """Surface remeshing bound to one existing source mesh identity."""

    surface: SurfaceMeshingSpec
    source_mesh_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    def __init__(self, surface: SurfaceMeshingSpec, source_mesh_id: str, /):
        if not isinstance(surface, SurfaceMeshingSpec):
            raise TypeError("surface must be SurfaceMeshingSpec.")
        mesh_id = str(source_mesh_id).strip()
        if not mesh_id:
            raise ValueError("source_mesh_id must be non-empty.")
        self.surface = surface
        self.source_mesh_id = mesh_id
        self.specification_id = canonical_fingerprint(
            {
                "kind": "surface-remeshing-spec",
                "surface": surface.specification_id,
                "source_mesh_id": mesh_id,
            }
        )


class VolumeMeshingSpec(StrictModule, NonTrainableState):
    target: CellMeshingTarget
    boundary_scope: MeshingScope
    fill_strategy: VolumeFillStrategy = eqx.field(static=True)
    size_controls: tuple[SizeControl, ...]
    protected_features: tuple[ProtectedFeature, ...]
    region_controls: tuple[RegionControl, ...]
    patch_controls: tuple[PatchControl, ...]
    region_seeds: tuple[RegionSeed, ...]
    hole_seeds: tuple[HoleSeed, ...]
    layer_controls: tuple[SweptLayerControl, ...]
    periodic_constraints: tuple[PeriodicConstraint, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: SizeCompliancePolicy
    limits: MeshingLimits
    deterministic: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: CellMeshingTarget,
        boundary_scope: MeshingScope,
        fill_strategy: VolumeFillStrategy,
        /,
        *,
        size_controls: tuple[SizeControl, ...],
        protected_features: tuple[ProtectedFeature, ...] = (),
        region_controls: tuple[RegionControl, ...] = (),
        patch_controls: tuple[PatchControl, ...] = (),
        region_seeds: tuple[RegionSeed, ...] = (),
        hole_seeds: tuple[HoleSeed, ...] = (),
        layer_controls: tuple[SweptLayerControl, ...] = (),
        periodic_constraints: tuple[PeriodicConstraint, ...] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: SizeCompliancePolicy | None = None,
        limits: MeshingLimits | None = None,
        deterministic: bool = True,
    ):
        if not isinstance(target, CellMeshingTarget) or target.topological_dimension != 3:
            raise ValueError(
                "Volume meshing target must have topological dimension three."
            )
        if not isinstance(fill_strategy, VolumeFillStrategy):
            raise TypeError("fill_strategy must be VolumeFillStrategy.")
        (
            sizes,
            features,
            regions,
            patches,
            periodic,
            compliance,
        ) = _validated_semantic_controls(
            target,
            boundary_scope,
            size_controls,
            protected_features,
            region_controls,
            patch_controls,
            periodic_constraints,
            size_combination,
            size_compliance,
        )
        region_seeds_ = tuple(region_seeds)
        hole_seeds_ = tuple(hole_seeds)
        layer_controls_ = tuple(layer_controls)
        if not all(isinstance(seed, RegionSeed) for seed in region_seeds_):
            raise TypeError("region_seeds must contain RegionSeed values.")
        if not all(isinstance(seed, HoleSeed) for seed in hole_seeds_):
            raise TypeError("hole_seeds must contain HoleSeed values.")
        if not all(isinstance(control, SweptLayerControl) for control in layer_controls_):
            raise TypeError("layer_controls must contain only SweptLayerControl values.")
        if any(
            control.volume_scope.entity_set_id != boundary_scope.entity_set_id
            or np.setdiff1d(
                np.asarray(control.volume_scope.entity_ids),
                np.asarray(boundary_scope.entity_ids),
            ).size
            for control in layer_controls_
        ):
            raise ValueError(
                "Swept-layer volume scopes must be contained in the top-level volume scope."
            )
        layer_scopes = tuple(
            scope
            for control in layer_controls_
            for scope in (
                control.source_scope,
                control.target_scope,
                control.volume_scope,
            )
        )
        binding = (
            boundary_scope.source_id,
            boundary_scope.source_revision,
            boundary_scope.entity_kind,
        )
        additional_scopes = (
            *(seed.scope for seed in hole_seeds_),
            *layer_scopes,
        )
        if any(
            (scope.source_id, scope.source_revision, scope.entity_kind) != binding
            for scope in additional_scopes
        ):
            raise ValueError("Volume meshing controls must share one source binding.")
        limit = MeshingLimits() if limits is None else limits
        if not isinstance(limit, MeshingLimits):
            raise TypeError("limits must be MeshingLimits or None.")
        self.target = target
        self.boundary_scope = boundary_scope
        self.fill_strategy = fill_strategy
        self.size_controls = sizes
        self.protected_features = features
        self.region_controls = regions
        self.patch_controls = patches
        self.region_seeds = region_seeds_
        self.hole_seeds = hole_seeds_
        self.layer_controls = layer_controls_
        self.periodic_constraints = periodic
        self.size_combination = size_combination
        self.size_compliance = compliance
        self.limits = limit
        self.deterministic = bool(deterministic)
        self.specification_id = canonical_fingerprint(
            {
                "kind": "volume-meshing-spec",
                "target": target.target_id,
                "boundary_scope": boundary_scope.scope_id,
                "fill_strategy": fill_strategy.value,
                "size_controls": [control.control_id for control in sizes],
                "size_combination": size_combination.value,
                "size_compliance": compliance.policy_id,
                "protected_features": [value.feature_id for value in features],
                "regions": [value.control_id for value in regions],
                "patches": [value.control_id for value in patches],
                "region_seeds": [value.seed_id for value in region_seeds_],
                "hole_seeds": [value.seed_id for value in hole_seeds_],
                "layers": [value.control_id for value in layer_controls_],
                "periodic": [value.constraint_id for value in periodic],
                "limits": limit.limits_id,
                "deterministic": bool(deterministic),
            }
        )


MeshingSpecification = SurfaceMeshingSpec | SurfaceRemeshingSpec | VolumeMeshingSpec


class MeshingProviderInfo(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    license_spdx: str = eqx.field(static=True)
    operations: tuple[MeshingOperation, ...] = eqx.field(static=True)
    source_kinds: tuple[MeshingSourceKind, ...] = eqx.field(static=True)
    capabilities: tuple[MeshingCapability, ...] = eqx.field(static=True)
    cell_kinds: tuple[str, ...] = eqx.field(static=True)
    dimensions: tuple[int, ...] = eqx.field(static=True)
    execution_modes: tuple[MeshingExecutionMode, ...] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        version: str,
        license_spdx: str,
        /,
        *,
        operations: tuple[MeshingOperation, ...],
        source_kinds: tuple[MeshingSourceKind, ...],
        capabilities: tuple[MeshingCapability, ...],
        cell_kinds: tuple[str, ...],
        dimensions: tuple[int, ...],
        execution_modes: tuple[MeshingExecutionMode, ...],
    ):
        values = tuple(str(value).strip() for value in (name, version, license_spdx))
        if any(not value for value in values):
            raise ValueError("Provider name, version, and license must be non-empty.")
        if (
            not operations
            or not source_kinds
            or not cell_kinds
            or not dimensions
            or not execution_modes
        ):
            raise ValueError("Provider support sets must be non-empty.")
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("Provider dimensions must be positive.")
        self.name, self.version, self.license_spdx = values
        self.operations = tuple(operations)
        self.source_kinds = tuple(source_kinds)
        self.capabilities = tuple(capabilities)
        self.cell_kinds = tuple(str(value) for value in cell_kinds)
        self.dimensions = tuple(dimensions)
        self.execution_modes = tuple(execution_modes)
        self.provider_id = canonical_fingerprint(
            {
                "kind": "meshing-provider-info",
                "name": values[0],
                "version": values[1],
                "license": values[2],
                "operations": [value.value for value in operations],
                "source_kinds": [value.value for value in source_kinds],
                "capabilities": [value.value for value in capabilities],
                "cell_kinds": self.cell_kinds,
                "dimensions": self.dimensions,
                "execution_modes": [value.value for value in execution_modes],
            }
        )


class MeshingSourceDescriptor(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_kind: MeshingSourceKind = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    source_descriptor_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        source_kind: MeshingSourceKind,
        topological_dimension: int,
        ambient_dimension: int,
        /,
        *,
        closed: bool,
    ):
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        topological = int(topological_dimension)
        ambient = int(ambient_dimension)
        if not source or not revision:
            raise ValueError("Meshing source identities must be non-empty.")
        if not isinstance(source_kind, MeshingSourceKind):
            raise TypeError("source_kind must be MeshingSourceKind.")
        if topological <= 0 or ambient < topological:
            raise ValueError("Source dimensions must satisfy 0 < topological <= ambient.")
        self.source_id = source
        self.source_revision = revision
        self.source_kind = source_kind
        self.topological_dimension = topological
        self.ambient_dimension = ambient
        self.closed = bool(closed)
        self.source_descriptor_id = canonical_fingerprint(
            {
                "kind": "meshing-source-descriptor",
                "source_id": source,
                "source_revision": revision,
                "source_kind": source_kind.value,
                "dimensions": [topological, ambient],
                "closed": bool(closed),
            }
        )


class ProviderSupportReport(StrictModule, NonTrainableState):
    provider_id: str = eqx.field(static=True)
    source_descriptor_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    supported: bool = eqx.field(static=True)
    unsupported: tuple[str, ...] = eqx.field(static=True)
    weakened_guarantees: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: MeshingProviderInfo,
        source: MeshingSourceDescriptor,
        specification: MeshingSpecification,
        /,
        *,
        unsupported: tuple[str, ...] = (),
        weakened_guarantees: tuple[str, ...] = (),
    ):
        if not isinstance(provider, MeshingProviderInfo):
            raise TypeError("provider must be MeshingProviderInfo.")
        if not isinstance(source, MeshingSourceDescriptor):
            raise TypeError("source must be MeshingSourceDescriptor.")
        if not isinstance(
            specification,
            (SurfaceMeshingSpec, SurfaceRemeshingSpec, VolumeMeshingSpec),
        ):
            raise TypeError("specification must be a meshing specification.")
        unsupported_ = tuple(str(value) for value in unsupported)
        weakened = tuple(str(value) for value in weakened_guarantees)
        self.provider_id = provider.provider_id
        self.source_descriptor_id = source.source_descriptor_id
        self.specification_id = specification.specification_id
        self.supported = not unsupported_
        self.unsupported = unsupported_
        self.weakened_guarantees = weakened
        self.report_id = canonical_fingerprint(
            {
                "kind": "provider-support-report",
                "provider": provider.provider_id,
                "source": source.source_descriptor_id,
                "specification": specification.specification_id,
                "unsupported": unsupported_,
                "weakened_guarantees": weakened,
            }
        )

    def require_supported(self, /) -> None:
        if not self.supported:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(self.unsupported),
                provider_code="preflight",
            )


class MeshingFailure(RuntimeError):
    def __init__(
        self,
        category: MeshingFailureCategory,
        message: str,
        /,
        *,
        provider_code: str = "",
        stage: str = "",
        entity_ids: tuple[int, ...] = (),
        locations: tuple[tuple[float, ...], ...] = (),
    ):
        if not isinstance(category, MeshingFailureCategory):
            raise TypeError("category must be MeshingFailureCategory.")
        text = str(message).strip()
        if not text:
            raise ValueError("Meshing failures require a message.")
        self.category = category
        self.provider_code = str(provider_code)
        self.stage = str(stage)
        self.entity_ids = tuple(entity_ids)
        self.locations = tuple(
            tuple(float(component) for component in point) for point in locations
        )
        super().__init__(text)


__all__ = [
    "CellFamilyPolicy",
    "CellMeshingTarget",
    "MeshingCapability",
    "MeshingDerivativeMode",
    "MeshingExecutionMode",
    "MeshingFailure",
    "MeshingFailureCategory",
    "MeshingLimits",
    "MeshingOperation",
    "MeshingProviderInfo",
    "MeshingSourceDescriptor",
    "MeshingSourceKind",
    "MeshingSpecification",
    "ProviderSupportReport",
    "SurfaceMeshingSpec",
    "SurfaceRemeshingSpec",
    "VolumeFillStrategy",
    "VolumeMeshingSpec",
]
