#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._periodic_constraint as periodic_constraint_owner
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._fingerprint import canonical_fingerprint
from .._meshcore import NativeExecutionLimits
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, positive_finite_float, positive_integer
from ..geometry._planar_embedding import PlanarEmbedding
from ..typing import checked, parse
from ._sizing import SizeCombinationPolicy


class MeshingOperation(StrEnum):
    FACET_GEOMETRY = "facet_geometry"
    MESH_CURVE = "mesh_curve"
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
    PIECEWISE_LINEAR = "piecewise_linear"
    MAPPED_REFERENCE = "mapped_reference"
    CURVE = "curve"


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


class MeshingLimits(StrictModule, NonTrainableState):
    """Hard resource budgets of one meshing request.

    Entity, connectivity, and data limits bound the published result.
    ``maximum_work_units`` bounds native construction steps (inserted Steiner or
    refinement vertices and root solves), ``maximum_cavity_cells`` the cells of
    one local cavity edit, ``maximum_geometry_queries`` the source-geometry
    evaluations, and ``maximum_scratch_bytes`` the temporary working memory of
    preparation and construction. Routes refuse before allocating beyond a
    budget and report every budget they cannot enforce.
    """

    maximum_vertices: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_faces: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_connectivity_entries: int = eqx.field(static=True)
    maximum_data_bytes: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)
    maximum_cavity_cells: int = eqx.field(static=True)
    maximum_geometry_queries: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
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
        maximum_work_units: int = 1_000_000_000,
        maximum_cavity_cells: int = 1_000_000,
        maximum_geometry_queries: int = 1_000_000_000,
        maximum_scratch_bytes: int = 8_000_000_000,
        maximum_wall_seconds: float = 3600.0,
    ) -> None:
        counts = {
            "maximum_vertices": maximum_vertices,
            "maximum_edges": maximum_edges,
            "maximum_faces": maximum_faces,
            "maximum_cells": maximum_cells,
            "maximum_connectivity_entries": maximum_connectivity_entries,
            "maximum_data_bytes": maximum_data_bytes,
            "maximum_work_units": maximum_work_units,
            "maximum_cavity_cells": maximum_cavity_cells,
            "maximum_geometry_queries": maximum_geometry_queries,
            "maximum_scratch_bytes": maximum_scratch_bytes,
        }
        counts = {name: positive_integer(value, name) for name, value in counts.items()}
        seconds = positive_finite_float(maximum_wall_seconds, "maximum_wall_seconds")
        self.maximum_vertices = counts["maximum_vertices"]
        self.maximum_edges = counts["maximum_edges"]
        self.maximum_faces = counts["maximum_faces"]
        self.maximum_cells = counts["maximum_cells"]
        self.maximum_connectivity_entries = counts["maximum_connectivity_entries"]
        self.maximum_data_bytes = counts["maximum_data_bytes"]
        self.maximum_work_units = counts["maximum_work_units"]
        self.maximum_cavity_cells = counts["maximum_cavity_cells"]
        self.maximum_geometry_queries = counts["maximum_geometry_queries"]
        self.maximum_scratch_bytes = counts["maximum_scratch_bytes"]
        self.maximum_wall_seconds = seconds
        self.limits_id = canonical_fingerprint(
            {
                "kind": "meshing-limits",
                "counts": [[name, counts[name]] for name in sorted(counts)],
                "maximum_wall_seconds": seconds,
            }
        )

    def native_limits(self) -> NativeExecutionLimits:
        """Lower the request's mechanical allowance without publication limits."""
        return NativeExecutionLimits(
            self.maximum_work_units,
            self.maximum_geometry_queries,
            self.maximum_cavity_cells,
            self.maximum_scratch_bytes,
            self.maximum_wall_seconds,
        )


type CellFamily = Literal[
    "interval",
    "triangle",
    "quadrilateral",
    "polygon",
    "tetrahedron",
    "hexahedron",
    "prism",
    "pyramid",
    "polyhedron",
]


class CellFamilyPolicy(StrictModule, NonTrainableState):
    required: tuple[CellFamily, ...] = eqx.field(static=True)
    preferred: tuple[CellFamily, ...] = eqx.field(static=True)
    allowed_transitions: tuple[CellFamily, ...] = eqx.field(static=True)
    allow_mixed: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        required: tuple[str, ...] = (),
        preferred: tuple[str, ...] = (),
        allowed_transitions: tuple[str, ...] = (),
        allow_mixed: bool = False,
    ) -> None:
        normalized_required = tuple(str(value) for value in required)
        normalized_preferred = tuple(str(value) for value in preferred)
        normalized_transitions = tuple(str(value) for value in allowed_transitions)
        values = (*normalized_required, *normalized_preferred, *normalized_transitions)
        if not normalized_required and not normalized_preferred:
            raise ValueError(
                "CellFamilyPolicy.families must contain at least one required or preferred cell family; got empty roles"
            )
        required_ = tuple(
            parse(value, CellFamily, "CellFamilyPolicy.required")
            for value in normalized_required
        )
        preferred_ = tuple(
            parse(value, CellFamily, "CellFamilyPolicy.preferred")
            for value in normalized_preferred
        )
        transitions = tuple(
            parse(value, CellFamily, "CellFamilyPolicy.allowed_transitions")
            for value in normalized_transitions
        )
        if len(set(required_)) != len(required_) or len(set(preferred_)) != len(
            preferred_
        ):
            raise ValueError(
                f"CellFamilyPolicy.families must be unique within each role; got required={required_!r}, preferred={preferred_!r}"
            )
        if not allow_mixed and len(set(values)) > 1:
            raise ValueError(
                f"CellFamilyPolicy.allow_mixed must be True for multiple cell families; got {allow_mixed!r}"
            )
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

    @checked
    def __init__(
        self,
        topological_dimension: int,
        ambient_dimension: int,
        cell_families: CellFamilyPolicy,
        /,
        *,
        geometry_order: int = 1,
        require_conforming: bool = True,
    ) -> None:
        topological = int(topological_dimension)
        ambient = int(ambient_dimension)
        order = int(geometry_order)
        if topological <= 0 or ambient < topological:
            raise ValueError(
                f"CellMeshingTarget.dimensions must satisfy 0 < topological <= ambient; got {(topological, ambient)!r}"
            )
        if order <= 0:
            raise ValueError(
                f"CellMeshingTarget.geometry_order must be positive; got {order!r}"
            )
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


class MeshQualityTarget(StrictModule, NonTrainableState):
    """Requested per-cell shape quality of generated simplices.

    ``minimum_angle`` is the smallest admissible interior corner angle in
    radians. A hard target is a failed request when unmet; a soft target is
    recorded as requested versus achieved evidence.
    """

    minimum_angle: float = eqx.field(static=True)
    hard: bool = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(self, *, minimum_angle: float, hard: bool = True) -> None:
        angle = float(minimum_angle)
        if not np.isfinite(angle) or angle <= 0.0 or angle >= np.pi / 3.0:
            raise ValueError(
                f"MeshQualityTarget.minimum_angle must be finite and strictly between zero and pi/3; got {angle!r}"
            )
        if not isinstance(hard, (bool, np.bool_)):
            raise TypeError(
                f"MeshQualityTarget.hard must be a bool; got {type(hard).__name__}"
            )
        self.minimum_angle = angle
        self.hard = bool(hard)
        self.target_id = canonical_fingerprint(
            {"kind": "mesh-quality-target", "minimum_angle": angle, "hard": self.hard}
        )


@final
class _SemanticScopeState(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    source_volume_regions: bool = eqx.field(static=True)


def _validate_semantic_scopes(
    target: CellMeshingTarget,
    scope: _dependency_phydrax_meshing__scope.MeshingScope,
    sizes: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...],
    features: tuple[_dependency_phydrax_meshing__controls.ProtectedFeature, ...],
    regions: tuple[_dependency_phydrax_meshing__controls.RegionControl, ...],
    patches: tuple[_dependency_phydrax_meshing__controls.PatchControl, ...],
    periodic: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    volume_boundary: bool,
    surface_volume_boundaries: bool,
) -> _SemanticScopeState:
    binding = (scope.source_id, scope.source_revision, scope.entity_kind)
    scoped = (
        *(
            size_scope
            for control in sizes
            for size_scope in _dependency_phydrax_meshing__sizing.control_scopes(control)
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
        raise ValueError(
            "MeshingSpec.controls must share the top-level source binding; got controls from another source or revision"
        )
    dimension = target.topological_dimension
    boundary_scoped_regions = volume_boundary and scope.entity_dimension == dimension - 1
    if regions and scope.entity_dimension != dimension and not boundary_scoped_regions:
        raise ValueError(
            f"MeshingSpec.region_controls must use a top-dimensional source scope; got scope_dimension={scope.entity_dimension}"
        )
    if any(
        not (
            (
                control.scope.entity_dimension == dimension
                and (
                    boundary_scoped_regions
                    or (
                        control.scope.entity_set_id == scope.entity_set_id
                        and bool(
                            jnp.all(
                                _dependency_phydrax_meshing__scope.contains_ids(
                                    scope.global_entity_ids,
                                    control.scope.global_entity_ids,
                                )
                            )
                        )
                    )
                )
            )
            or (
                surface_volume_boundaries
                and scope.entity_kind
                is _dependency_phydrax_meshing__scope.MeshingEntityKind.GEOMETRY
                and control.scope.entity_dimension == dimension + 1
            )
        )
        for control in regions
    ):
        raise ValueError(
            f"MeshingSpec.region_controls must select contained top-dimensional entities; got {len(regions)} controls including an incompatible scope"
        )
    source_volume_regions = (
        surface_volume_boundaries
        and scope.entity_kind
        is _dependency_phydrax_meshing__scope.MeshingEntityKind.GEOMETRY
        and any(control.scope.entity_dimension == dimension + 1 for control in regions)
    )
    return _SemanticScopeState(dimension, source_volume_regions)


def _validate_semantic_patch_dimensions(
    scope: _dependency_phydrax_meshing__scope.MeshingScope,
    patches: tuple[_dependency_phydrax_meshing__controls.PatchControl, ...],
    dimension: int,
    source_volume_regions: bool,
) -> None:
    if any(
        control.scope.entity_dimension != dimension - 1
        and not (
            source_volume_regions
            and control.scope.entity_dimension == dimension
            and control.scope.entity_set_id == scope.entity_set_id
            and bool(
                jnp.all(
                    _dependency_phydrax_meshing__scope.contains_ids(
                        scope.global_entity_ids,
                        control.scope.global_entity_ids,
                    )
                )
            )
        )
        for control in patches
    ):
        if source_volume_regions:
            raise ValueError(
                f"MeshingSpec.patch_controls must select codimension-one entities or declared source-volume boundary faces; got {len(patches)} controls including an incompatible scope"
            )
        raise ValueError(
            f"MeshingSpec.patch_controls must select codimension-one entities; got {len(patches)} controls including an incompatible scope"
        )


def _validate_semantic_region_partition(
    regions: tuple[_dependency_phydrax_meshing__controls.RegionControl, ...],
) -> tuple[str, ...]:
    region_names = tuple(control.region_name for control in regions)
    if len(set(region_names)) != len(region_names):
        raise ValueError(
            f"MeshingSpec.region_controls.names must be unique; got {region_names!r}"
        )
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            if (
                first.scope.entity_set_id == second.scope.entity_set_id
                and _dependency_phydrax_meshing__scope.overlapping_ids(
                    first.scope.global_entity_ids, second.scope.global_entity_ids
                )
            ):
                raise ValueError(
                    f"MeshingSpec.region_controls.scopes must be disjoint; got overlapping controls {first.region_name!r} and {second.region_name!r}"
                )
    return region_names


def _validate_semantic_patch_partition(
    patches: tuple[_dependency_phydrax_meshing__controls.PatchControl, ...],
    region_names: tuple[str, ...],
) -> None:
    patch_names = tuple(control.name for control in patches)
    if len(set(patch_names)) != len(patch_names):
        raise ValueError(
            f"MeshingSpec.patch_controls.names must be unique; got {patch_names!r}"
        )
    for index, first in enumerate(patches):
        for second in patches[index + 1 :]:
            if first.scope.entity_set_id == second.scope.entity_set_id and bool(
                jnp.any(
                    _dependency_phydrax_meshing__scope.contains_ids(
                        first.scope.global_entity_ids,
                        second.scope.global_entity_ids,
                    )
                )
            ):
                raise ValueError(
                    f"MeshingSpec.patch_controls.scopes must be disjoint; got overlapping controls {first.name!r} and {second.name!r}"
                )
    declared = set(region_names)
    if any(not set(control.adjacent_region_names) <= declared for control in patches):
        raise ValueError(
            f"MeshingSpec.patch_controls must reference declared region names; got references outside {region_names!r}"
        )


# Keep kind admission fail-closed before ordered scope, region, and patch validation.
def _validated_semantic_controls(
    target: CellMeshingTarget,
    scope: _dependency_phydrax_meshing__scope.MeshingScope,
    size_controls: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...],
    protected_features: tuple[
        _dependency_phydrax_meshing__controls.ProtectedFeature, ...
    ],
    region_controls: tuple[_dependency_phydrax_meshing__controls.RegionControl, ...],
    patch_controls: tuple[_dependency_phydrax_meshing__controls.PatchControl, ...],
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    size_combination: SizeCombinationPolicy,
    size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy | None,
    /,
    *,
    volume_boundary: bool = False,
    surface_volume_boundaries: bool = False,
) -> tuple[
    tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...],
    tuple[_dependency_phydrax_meshing__controls.ProtectedFeature, ...],
    tuple[_dependency_phydrax_meshing__controls.RegionControl, ...],
    tuple[_dependency_phydrax_meshing__controls.PatchControl, ...],
    tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    _dependency_phydrax_meshing__sizing.SizeCompliancePolicy,
]:
    expected_scope_dimension = (
        target.topological_dimension - 1
        if target.topological_dimension == 3 and scope.entity_dimension == 2
        else target.topological_dimension
    )
    if scope.entity_dimension != expected_scope_dimension:
        raise ValueError(
            f"MeshingSpec.scope must have dimension {expected_scope_dimension}; got {scope.entity_dimension}"
        )
    sizes = tuple(size_controls)
    features = tuple(protected_features)
    regions = tuple(region_controls)
    patches = tuple(patch_controls)
    periodic = tuple(periodic_constraints)
    if not sizes:
        raise ValueError(
            "MeshingSpec.size_controls must contain at least one size control; got an empty sequence"
        )
    if not all(
        isinstance(control, _dependency_phydrax_meshing__sizing.SizeControl)
        for control in sizes
    ):
        raise TypeError(
            f"MeshingSpec.size_controls must contain SizeControl values; got {tuple(type(control).__name__ for control in sizes)!r}"
        )
    if not all(
        isinstance(feature, _dependency_phydrax_meshing__controls.ProtectedFeature)
        for feature in features
    ):
        raise TypeError(
            f"MeshingSpec.protected_features must contain ProtectedFeature values; got {tuple(type(feature).__name__ for feature in features)!r}"
        )
    if not all(
        isinstance(control, _dependency_phydrax_meshing__controls.RegionControl)
        for control in regions
    ):
        raise TypeError(
            f"MeshingSpec.region_controls must contain RegionControl values; got {tuple(type(control).__name__ for control in regions)!r}"
        )
    if not all(
        isinstance(control, _dependency_phydrax_meshing__controls.PatchControl)
        for control in patches
    ):
        raise TypeError(
            f"MeshingSpec.patch_controls must contain PatchControl values; got {tuple(type(control).__name__ for control in patches)!r}"
        )
    if not all(
        isinstance(constraint, periodic_constraint_owner.PeriodicConstraint)
        for constraint in periodic
    ):
        raise TypeError(
            f"MeshingSpec.periodic_constraints must contain PeriodicConstraint values; got {tuple(type(constraint).__name__ for constraint in periodic)!r}"
        )
    size_combination = parse(
        size_combination, SizeCombinationPolicy, "MeshingSpec.size_combination"
    )
    compliance = (
        _dependency_phydrax_meshing__sizing.SizeCompliancePolicy()
        if size_compliance is None
        else size_compliance
    )

    scoped_state = _validate_semantic_scopes(
        target,
        scope,
        sizes,
        features,
        regions,
        patches,
        periodic,
        volume_boundary,
        surface_volume_boundaries,
    )
    _validate_semantic_patch_dimensions(
        scope, patches, scoped_state.dimension, scoped_state.source_volume_regions
    )
    region_names = _validate_semantic_region_partition(regions)
    _validate_semantic_patch_partition(patches, region_names)
    return sizes, features, regions, patches, periodic, compliance


class SurfaceMeshingSpec(StrictModule, NonTrainableState):
    """Physical surface request with explicit source-region boundary semantics.

    Geometry-scoped dimension-three region controls select adjacent
    source volumes, not volume cells in the dimension-two result. Their boundary
    labels and oriented patches require source-incidence validation by the compiler.
    Dimension-two region controls retain ordinary contained surface-region semantics.
    """

    target: CellMeshingTarget
    planar_embedding: PlanarEmbedding | None = eqx.field(static=True)
    scope: _dependency_phydrax_meshing__scope.MeshingScope
    size_controls: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...]
    protected_features: tuple[_dependency_phydrax_meshing__controls.ProtectedFeature, ...]
    region_controls: tuple[_dependency_phydrax_meshing__controls.RegionControl, ...]
    patch_controls: tuple[_dependency_phydrax_meshing__controls.PatchControl, ...]
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...]
    layer_controls: tuple[_dependency_phydrax_meshing__controls.BoundaryLayerControl, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
    limits: MeshingLimits
    quality_target: MeshQualityTarget | None
    background_metric: (
        _dependency_phydrax_meshing__controls.BackgroundMetricControl | None
    )
    deterministic: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        target: CellMeshingTarget,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        /,
        *,
        planar_embedding: PlanarEmbedding | None = None,
        size_controls: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...],
        protected_features: tuple[
            _dependency_phydrax_meshing__controls.ProtectedFeature, ...
        ] = (),
        region_controls: tuple[
            _dependency_phydrax_meshing__controls.RegionControl, ...
        ] = (),
        patch_controls: tuple[
            _dependency_phydrax_meshing__controls.PatchControl, ...
        ] = (),
        periodic_constraints: tuple[
            periodic_constraint_owner.PeriodicConstraint, ...
        ] = (),
        layer_controls: tuple[
            _dependency_phydrax_meshing__controls.BoundaryLayerControl, ...
        ] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
        | None = None,
        limits: MeshingLimits | None = None,
        quality_target: MeshQualityTarget | None = None,
        background_metric: _dependency_phydrax_meshing__controls.BackgroundMetricControl
        | None = None,
        deterministic: bool = True,
    ) -> None:
        if target.topological_dimension != 2:
            raise ValueError(
                f"SurfaceMeshingSpec.target must have topological dimension two; got {target.topological_dimension}"
            )
        if (target.ambient_dimension == 2) != (planar_embedding is not None):
            raise ValueError(
                f"SurfaceMeshingSpec.planar_embedding must be supplied exactly for ambient-dimension-two surface targets; got ambient_dimension={target.ambient_dimension}, embedding_present={planar_embedding is not None}"
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
            surface_volume_boundaries=True,
        )
        layers = tuple(layer_controls)
        if not all(
            isinstance(
                control, _dependency_phydrax_meshing__controls.BoundaryLayerControl
            )
            for control in layers
        ):
            raise TypeError(
                f"MeshingSpec.layer_controls must contain only BoundaryLayerControl values; got {tuple(type(control).__name__ for control in layer_controls)!r}"
            )
        if any(
            control.wall_scope.entity_dimension != 1
            or (
                control.wall_scope.source_id,
                control.wall_scope.source_revision,
                control.wall_scope.entity_kind,
            )
            != (scope.source_id, scope.source_revision, scope.entity_kind)
            for control in layers
        ):
            raise ValueError(
                f"SurfaceMeshingSpec.layer_controls must grow from wall curves of the meshed source; got {len(layers)} controls including an incompatible wall scope"
            )
        limit = MeshingLimits() if limits is None else limits
        if background_metric is not None:
            if (
                background_metric.mesh.ambient_dimension != target.ambient_dimension
                or background_metric.metric.values.shape[1:]
                != (target.ambient_dimension, target.ambient_dimension)
            ):
                raise ValueError(
                    f"SurfaceMeshingSpec.background_metric must use the surface physical ambient space; got mesh_dimension={background_metric.mesh.ambient_dimension}, tensor_shape={background_metric.metric.values.shape[1:]!r}"
                )
        self.target = target
        self.planar_embedding = planar_embedding
        self.scope = scope
        self.size_controls = sizes
        self.protected_features = features
        self.region_controls = regions
        self.patch_controls = patches
        self.periodic_constraints = periodic
        self.layer_controls = layers
        self.size_combination = size_combination
        self.size_compliance = compliance
        self.limits = limit
        self.quality_target = quality_target
        self.background_metric = background_metric
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
                "layers": [value.control_id for value in layers],
                "limits": limit.limits_id,
                "quality_target": (
                    None if quality_target is None else quality_target.target_id
                ),
                "background_metric": (
                    None if background_metric is None else background_metric.control_id
                ),
                "deterministic": bool(deterministic),
            }
        )


class SurfaceRemeshingSpec(StrictModule, NonTrainableState):
    """Surface remeshing bound to one existing source mesh identity."""

    surface: SurfaceMeshingSpec
    source_mesh_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    @checked
    def __init__(self, surface: SurfaceMeshingSpec, source_mesh_id: str, /) -> None:
        mesh_id = str(source_mesh_id).strip()
        if not mesh_id:
            raise ValueError(
                f"SurfaceRemeshingSpec.source_mesh_id must be non-empty; got {mesh_id!r}"
            )
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
    """Physical volume request with separately declared boundary and region scopes.

    A codimension-one boundary scope does not contain volume-region entities.
    Region controls select top-dimensional entities of the same declared
    source revision; the provider resolves their scientific region identities.
    """

    target: CellMeshingTarget
    boundary_scope: _dependency_phydrax_meshing__scope.MeshingScope
    fill_strategy: VolumeFillStrategy = eqx.field(static=True)
    size_controls: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...]
    protected_features: tuple[_dependency_phydrax_meshing__controls.ProtectedFeature, ...]
    region_controls: tuple[_dependency_phydrax_meshing__controls.RegionControl, ...]
    patch_controls: tuple[_dependency_phydrax_meshing__controls.PatchControl, ...]
    region_seeds: tuple[_dependency_phydrax_meshing__controls.RegionSeed, ...]
    hole_seeds: tuple[_dependency_phydrax_meshing__controls.HoleSeed, ...]
    layer_controls: tuple[_dependency_phydrax_meshing__controls.BoundaryLayerControl, ...]
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
    limits: MeshingLimits
    deterministic: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True, init=False)

    @checked
    def __init__(
        self,
        target: CellMeshingTarget,
        boundary_scope: _dependency_phydrax_meshing__scope.MeshingScope,
        fill_strategy: VolumeFillStrategy,
        *,
        size_controls: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...],
        protected_features: tuple[
            _dependency_phydrax_meshing__controls.ProtectedFeature, ...
        ] = (),
        region_controls: tuple[
            _dependency_phydrax_meshing__controls.RegionControl, ...
        ] = (),
        patch_controls: tuple[
            _dependency_phydrax_meshing__controls.PatchControl, ...
        ] = (),
        region_seeds: tuple[_dependency_phydrax_meshing__controls.RegionSeed, ...] = (),
        hole_seeds: tuple[_dependency_phydrax_meshing__controls.HoleSeed, ...] = (),
        layer_controls: tuple[
            _dependency_phydrax_meshing__controls.BoundaryLayerControl, ...
        ] = (),
        periodic_constraints: tuple[
            periodic_constraint_owner.PeriodicConstraint, ...
        ] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
        | None = None,
        limits: MeshingLimits | None = None,
        deterministic: bool = True,
    ) -> None:
        if target.topological_dimension != 3:
            raise ValueError(
                f"VolumeMeshingSpec.target must have topological dimension three; got {target.topological_dimension}"
            )
        fill_strategy = parse(
            fill_strategy, VolumeFillStrategy, "VolumeMeshingSpec.fill_strategy"
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
            boundary_scope,
            size_controls,
            protected_features,
            region_controls,
            patch_controls,
            periodic_constraints,
            size_combination,
            size_compliance,
            volume_boundary=True,
        )
        region_seeds_ = tuple(region_seeds)
        hole_seeds_ = tuple(hole_seeds)
        layer_controls_ = tuple(layer_controls)
        if not all(
            isinstance(seed, _dependency_phydrax_meshing__controls.RegionSeed)
            for seed in region_seeds_
        ):
            raise TypeError(
                f"VolumeMeshingSpec.region_seeds must contain RegionSeed values; got {tuple(type(seed).__name__ for seed in region_seeds_)!r}"
            )
        if not all(
            isinstance(seed, _dependency_phydrax_meshing__controls.HoleSeed)
            for seed in hole_seeds_
        ):
            raise TypeError(
                f"VolumeMeshingSpec.hole_seeds must contain HoleSeed values; got {tuple(type(seed).__name__ for seed in hole_seeds_)!r}"
            )
        if not all(
            isinstance(
                control, _dependency_phydrax_meshing__controls.BoundaryLayerControl
            )
            for control in layer_controls_
        ):
            raise TypeError(
                f"MeshingSpec.layer_controls must contain only BoundaryLayerControl values; got {tuple(type(control).__name__ for control in layer_controls)!r}"
            )
        if any(
            control.volume_scope is None
            or control.volume_scope.entity_dimension != target.topological_dimension
            for control in layer_controls_
        ):
            raise ValueError(
                f"VolumeMeshingSpec.layer_controls.volume_scope must select top-dimensional source region entities; got {len(layer_controls_)} controls including an incompatible scope"
            )
        layer_scopes = tuple(
            scope
            for control in layer_controls_
            for scope in (control.wall_scope, control.cap_scope, control.volume_scope)
            if scope is not None
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
            raise ValueError(
                f"VolumeMeshingSpec.controls must share one source binding; got controls outside {binding!r}"
            )
        if boundary_scope.entity_dimension == target.topological_dimension and any(
            scope.entity_dimension == target.topological_dimension
            and (
                scope.entity_set_id != boundary_scope.entity_set_id
                or not bool(
                    np.all(
                        np.isin(
                            np.asarray(scope.global_entity_ids),
                            np.asarray(boundary_scope.global_entity_ids),
                        )
                    )
                )
            )
            for scope in layer_scopes
        ):
            raise ValueError(
                f"VolumeMeshingSpec.layer_controls.volume_scope must select contained top-dimensional source entities; got {len(layer_scopes)} scopes including a noncontained scope"
            )
        if any(
            control.wall_scope.entity_dimension != target.topological_dimension - 1
            or (
                boundary_scope.entity_dimension == target.topological_dimension - 1
                and (
                    control.wall_scope.entity_set_id != boundary_scope.entity_set_id
                    or not bool(
                        np.all(
                            np.isin(
                                np.asarray(control.wall_scope.global_entity_ids),
                                np.asarray(boundary_scope.global_entity_ids),
                            )
                        )
                    )
                )
            )
            for control in layer_controls_
        ):
            raise ValueError(
                f"VolumeMeshingSpec.layer_controls.wall_scope must select contained boundary entities; got {len(layer_controls_)} controls including a noncontained scope"
            )
        limit = MeshingLimits() if limits is None else limits
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


CurveEnd: TypeAlias = Literal["start", "end"]


class CurveJunction(StrictModule, NonTrainableState):
    """One declared junction joining curve endpoints into one mesh vertex.

    ``endpoints`` name ``(curve entity ID, end)`` pairs of the meshed source.
    A junction joining the start and end of one curve closes that curve; three
    or more endpoints declare a network junction. Endpoints are never joined
    by proximity: an undeclared endpoint remains a free end.
    """

    name: str = eqx.field(static=True)
    endpoints: tuple[tuple[int, CurveEnd], ...] = eqx.field(static=True)
    junction_id: str = eqx.field(static=True)

    def __init__(self, name: str, endpoints: tuple[tuple[int, CurveEnd], ...], /) -> None:
        value = str(name).strip()
        if not value:
            raise ValueError(f"CurveJunction.name must be non-empty; got {value!r}")
        entries = tuple(endpoints)
        if len(entries) < 2:
            raise ValueError(
                f"CurveJunction.endpoints must join at least two curve endpoints; got {len(entries)}"
            )
        normalized = []
        for entry in entries:
            if len(entry) != 2:
                raise ValueError(
                    f"CurveJunction.endpoints must contain (curve ID, end) pairs; got entry_length={len(entry)}"
                )
            curve, end = entry
            curve = nonnegative_integer(curve, "CurveJunction.curve")
            normalized.append((int(curve), parse(end, CurveEnd, "end")))
        ordered = tuple(sorted(normalized))
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"CurveJunction.endpoints must be unique; got {ordered!r}")
        self.name = value
        self.endpoints = ordered
        self.junction_id = canonical_fingerprint(
            {
                "kind": "curve-junction",
                "name": value,
                "endpoints": [[curve, end] for curve, end in ordered],
            }
        )


class CurveMeshingSpec(StrictModule, NonTrainableState):
    """Standalone interval meshing of selected source curves and their network.

    ``scope`` selects dimension-one source curves; ``junctions`` declare which
    curve endpoints share a mesh vertex. Size controls bound interval lengths,
    and a protected ``FeatureKind.CURVE`` feature bounds the chord deviation of
    its curves by ``maximum_deviation``.
    """

    target: CellMeshingTarget
    scope: _dependency_phydrax_meshing__scope.MeshingScope
    size_controls: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...]
    protected_features: tuple[_dependency_phydrax_meshing__controls.ProtectedFeature, ...]
    junctions: tuple[CurveJunction, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
    limits: MeshingLimits
    deterministic: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        target: CellMeshingTarget,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        /,
        *,
        size_controls: tuple[_dependency_phydrax_meshing__sizing.SizeControl, ...],
        protected_features: tuple[
            _dependency_phydrax_meshing__controls.ProtectedFeature, ...
        ] = (),
        junctions: tuple[CurveJunction, ...] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
        | None = None,
        limits: MeshingLimits | None = None,
        deterministic: bool = True,
    ) -> None:
        if target.topological_dimension != 1:
            raise ValueError(
                f"CurveMeshingSpec.target must have topological dimension one; got {target.topological_dimension}"
            )
        families = target.cell_families
        if set((*families.required, *families.preferred)) != {"interval"}:
            raise ValueError(
                f"CurveMeshingSpec.target.cell_families must select the interval cell family only; got required={families.required!r}, preferred={families.preferred!r}"
            )
        (
            sizes,
            features,
            _,
            _,
            _,
            compliance,
        ) = _validated_semantic_controls(
            target,
            scope,
            size_controls,
            protected_features,
            (),
            (),
            (),
            size_combination,
            size_compliance,
        )
        if any(
            feature.feature_kind
            is _dependency_phydrax_meshing__controls.FeatureKind.SURFACE
            for feature in features
        ):
            raise ValueError(
                f"CurveMeshingSpec.protected_features must not select surface features; got {len(features)} features including a surface scope"
            )
        junctions_ = tuple(junctions)
        if not all(isinstance(junction, CurveJunction) for junction in junctions_):
            raise TypeError(
                f"CurveMeshingSpec.junctions must contain CurveJunction values; got {tuple(type(junction).__name__ for junction in junctions_)!r}"
            )
        names = tuple(junction.name for junction in junctions_)
        if len(set(names)) != len(names):
            raise ValueError(
                f"CurveMeshingSpec.junctions.names must be unique; got {names!r}"
            )
        endpoints = tuple(
            endpoint for junction in junctions_ for endpoint in junction.endpoints
        )
        if len(set(endpoints)) != len(endpoints):
            raise ValueError(
                f"CurveMeshingSpec.junctions.endpoints must belong to at most one junction; got {endpoints!r}"
            )
        curves = np.asarray(tuple((curve for curve, _ in endpoints)), dtype=np.int64)
        if not bool(
            np.all(np.isin(np.asarray(curves), np.asarray(scope.global_entity_ids)))
        ):
            raise ValueError(
                f"CurveMeshingSpec.junctions must join curves of the meshing scope; got {curves.tolist()!r}"
            )
        ordered = tuple(sorted(junctions_, key=lambda junction: junction.endpoints))
        limit = MeshingLimits() if limits is None else limits
        self.target = target
        self.scope = scope
        self.size_controls = sizes
        self.protected_features = features
        self.junctions = ordered
        self.size_combination = size_combination
        self.size_compliance = compliance
        self.limits = limit
        self.deterministic = bool(deterministic)
        self.specification_id = canonical_fingerprint(
            {
                "kind": "curve-meshing-spec",
                "target": target.target_id,
                "scope": scope.scope_id,
                "size_controls": [control.control_id for control in sizes],
                "size_combination": size_combination.value,
                "size_compliance": compliance.policy_id,
                "protected_features": [value.feature_id for value in features],
                "junctions": [value.junction_id for value in ordered],
                "limits": limit.limits_id,
                "deterministic": bool(deterministic),
            }
        )


MeshingSpecification = (
    CurveMeshingSpec | SurfaceMeshingSpec | SurfaceRemeshingSpec | VolumeMeshingSpec
)


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
    ) -> None:
        values = tuple(str(value).strip() for value in (name, version, license_spdx))
        if any(not value for value in values):
            raise ValueError(
                f"MeshingProviderInfo.identifiers must be non-empty; got {values!r}"
            )
        if (
            not operations
            or not source_kinds
            or not cell_kinds
            or not dimensions
            or not execution_modes
        ):
            raise ValueError(
                f"MeshingProviderInfo.support must have non-empty operations, source kinds, cell kinds, dimensions and execution modes; got counts={(len(operations), len(source_kinds), len(cell_kinds), len(dimensions), len(execution_modes))!r}"
            )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError(
                f"MeshingProviderInfo.dimensions must be positive; got {dimensions!r}"
            )
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
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        topological = int(topological_dimension)
        ambient = int(ambient_dimension)
        if not source or not revision:
            raise ValueError(
                f"MeshingSourceDescriptor.identifiers must be non-empty; got {(source, revision)!r}"
            )
        source_kind = parse(
            source_kind, MeshingSourceKind, "MeshingSourceDescriptor.source_kind"
        )
        if topological <= 0 or ambient < topological:
            raise ValueError(
                f"MeshingSourceDescriptor.dimensions must satisfy 0 < topological <= ambient; got {(topological, ambient)!r}"
            )
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

    @checked
    def __init__(
        self,
        provider: MeshingProviderInfo,
        source: MeshingSourceDescriptor,
        specification: MeshingSpecification,
        /,
        *,
        unsupported: tuple[str, ...] = (),
        weakened_guarantees: tuple[str, ...] = (),
    ) -> None:
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
                f"ProviderSupportReport.specification must have supported controls; got unsupported controls={self.unsupported!r}",
                provider_code="preflight",
            )


__all__ = [
    "CellFamily",
    "CellFamilyPolicy",
    "CellMeshingTarget",
    "CurveEnd",
    "CurveJunction",
    "CurveMeshingSpec",
    "MeshQualityTarget",
    "MeshingCapability",
    "MeshingDerivativeMode",
    "MeshingExecutionMode",
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
