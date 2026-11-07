#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, nonnegative_integer, normalized_identifier
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry.brep._model import BRepModel
from ..typing import checked, parse
from ._controls import (
    BackgroundMetricControl,
    BoundaryLayerControl,
    FeatureKind,
    HoleSeed,
    PatchControl,
    PeriodicConstraint,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
)
from ._dual_extraction import SurfaceCrossField
from ._scope import MeshingScope
from ._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    SizeCombinationPolicy,
    SizeCompliancePolicy,
    SizeControl,
    UniformSizeControl,
)


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


@dataclass(frozen=True, slots=True)
class MeshingFailureEvidence:
    """Requested and achieved values retained with one failed meshing operation."""

    requested_quantities: tuple[tuple[str, float], ...]
    achieved_quantities: tuple[tuple[str, float], ...]
    checkpoint: str
    logical_findings: tuple[str, ...]
    evidence_id: str = field(init=False)

    def __init__(
        self,
        *,
        requested_quantities: Mapping[str, float] | Sequence[tuple[str, float]] = (),
        achieved_quantities: Mapping[str, float] | Sequence[tuple[str, float]] = (),
        checkpoint: str = "",
        logical_findings: Sequence[str] = (),
    ) -> None:
        def quantities(
            values: Mapping[str, float] | Sequence[tuple[str, float]],
            name: str,
        ) -> tuple[tuple[str, float], ...]:
            entries = values.items() if isinstance(values, Mapping) else values
            records: list[tuple[str, float]] = []
            names: set[str] = set()
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise ValueError(
                        f"MeshingFailureEvidence.{name} must contain name/value pairs."
                    )
                key = normalized_identifier(entry[0], f"{name}.name")
                if key in names:
                    raise ValueError(
                        f"MeshingFailureEvidence.{name} quantity names must be unique."
                    )
                names.add(key)
                value = finite_real_scalar(entry[1], f"{name}.{key}")
                records.append((key, value))
            return tuple(sorted(records))

        if not isinstance(checkpoint, str):
            raise TypeError("MeshingFailureEvidence.checkpoint must be a string.")
        if not isinstance(logical_findings, Sequence) or isinstance(
            logical_findings, (str, bytes)
        ):
            raise TypeError(
                "MeshingFailureEvidence.logical_findings must be a sequence of strings."
            )
        findings = tuple(
            normalized_identifier(value, "logical finding") for value in logical_findings
        )
        requested = quantities(requested_quantities, "requested_quantities")
        achieved = quantities(achieved_quantities, "achieved_quantities")
        checkpoint_ = checkpoint.strip()
        object.__setattr__(self, "requested_quantities", requested)
        object.__setattr__(self, "achieved_quantities", achieved)
        object.__setattr__(self, "checkpoint", checkpoint_)
        object.__setattr__(self, "logical_findings", findings)
        object.__setattr__(
            self,
            "evidence_id",
            canonical_fingerprint(
                {
                    "kind": "meshing-failure-evidence",
                    "requested_quantities": requested,
                    "achieved_quantities": achieved,
                    "checkpoint": checkpoint_,
                    "logical_findings": findings,
                }
            ),
        )


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
    ) -> None:
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CellFamilyPolicy], value: CellFamilyPolicy, /
    ) -> CellFamilyPolicy:
        if type(value) is not cls:
            raise TypeError("value must be CellFamilyPolicy.")
        if type(value.allow_mixed) is not bool:
            raise TypeError("Stored cell-family policy allow_mixed must be bool.")
        rebuilt = cls(
            required=value.required,
            preferred=value.preferred,
            allowed_transitions=value.allowed_transitions,
            allow_mixed=value.allow_mixed,
        )
        if (
            rebuilt.policy_id != value.policy_id
            or rebuilt.required != value.required
            or rebuilt.preferred != value.preferred
            or rebuilt.allowed_transitions != value.allowed_transitions
            or rebuilt.allow_mixed != value.allow_mixed
        ):
            raise ValueError(
                "Stored cell-family policy differs from its canonical fields."
            )
        return rebuilt


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
                "Meshing dimensions must satisfy 0 < topological <= ambient."
            )
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CellMeshingTarget], value: CellMeshingTarget, /
    ) -> CellMeshingTarget:
        if type(value) is not cls:
            raise TypeError("value must be CellMeshingTarget.")
        if any(
            type(field) is not int
            for field in (
                value.topological_dimension,
                value.ambient_dimension,
                value.geometry_order,
            )
        ):
            raise TypeError(
                "Stored cell-meshing target dimensions and order must be ints."
            )
        if type(value.require_conforming) is not bool:
            raise TypeError("Stored cell-meshing target require_conforming must be bool.")
        cell_families = CellFamilyPolicy.from_stored_fields(value.cell_families)
        rebuilt = cls(
            value.topological_dimension,
            value.ambient_dimension,
            cell_families,
            geometry_order=value.geometry_order,
            require_conforming=value.require_conforming,
        )
        if (
            rebuilt.target_id != value.target_id
            or rebuilt.topological_dimension != value.topological_dimension
            or rebuilt.ambient_dimension != value.ambient_dimension
            or rebuilt.geometry_order != value.geometry_order
            or rebuilt.require_conforming != value.require_conforming
        ):
            raise ValueError(
                "Stored cell-meshing target differs from its canonical fields."
            )
        return rebuilt


def _size_control_scopes(control: SizeControl, /) -> tuple[MeshingScope, ...]:
    if isinstance(control, ProximitySizeControl):
        return control.source_scope, control.target_scope
    return (control.scope,)


def _validated_semantic_controls(
    target: CellMeshingTarget,
    scope: MeshingScope,
    size_controls: tuple[SizeControl, ...],
    protected_features: tuple[ProtectedFeature, ...],
    region_controls: tuple[RegionControl, ...],
    patch_controls: tuple[PatchControl, ...],
    periodic_constraints: tuple[PeriodicConstraint, ...],
    size_combination: SizeCombinationPolicy,
    size_compliance: SizeCompliancePolicy | None,
    /,
) -> tuple[
    tuple[SizeControl, ...],
    tuple[ProtectedFeature, ...],
    tuple[RegionControl, ...],
    tuple[PatchControl, ...],
    tuple[PeriodicConstraint, ...],
    SizeCompliancePolicy,
]:
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
    if any(control.scope.entity_dimension != dimension for control in regions):
        raise ValueError("Region controls require a top-dimensional source scope.")
    region_entity_set = (
        scope.entity_set_id
        if scope.entity_dimension == dimension
        else regions[0].scope.entity_set_id
        if regions
        else None
    )
    if any(
        control.scope.entity_set_id != region_entity_set
        or (
            scope.entity_dimension == dimension
            and np.setdiff1d(control.scope.entity_ids, scope.entity_ids).size
        )
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


def _source_deviation(value: float | None, /) -> float | None:
    """Keep unspecified, exact, and bounded source accuracy distinct."""
    if value is None:
        return None
    deviation = finite_real_scalar(value, "maximum_deviation")
    if deviation < 0.0:
        raise ValueError("maximum_deviation must be non-negative.")
    return deviation


class MeshQualityTarget(StrictModule, NonTrainableState):
    """Requested minimum physical corner angle in radians.

    Supported surface routes measure simplex corner angles or structured
    coordinate-map corner tangents, with explicit hard/soft semantics. This
    target does not bound interior map distortion or volume dihedral angles.
    """

    minimum_angle: float = eqx.field(static=True)
    hard: bool = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(self, *, minimum_angle: float, hard: bool = True) -> None:
        angle = finite_real_scalar(minimum_angle, "minimum_angle")
        if not 0.0 < angle < np.pi / 3.0:
            raise ValueError("minimum_angle must lie strictly between zero and pi/3.")
        if not isinstance(hard, (bool, np.bool_)):
            raise TypeError("hard must be a boolean.")
        self.minimum_angle = angle
        self.hard = bool(hard)
        self.target_id = canonical_fingerprint(
            {"kind": "mesh-quality-target", "minimum_angle": angle, "hard": self.hard}
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MeshQualityTarget], value: MeshQualityTarget, /
    ) -> MeshQualityTarget:
        if type(value) is not cls:
            raise TypeError("value must be MeshQualityTarget.")
        if type(value.minimum_angle) is not float or type(value.hard) is not bool:
            raise TypeError("Stored mesh-quality target fields are not canonical.")
        rebuilt = cls(minimum_angle=value.minimum_angle, hard=value.hard)
        if (
            rebuilt.target_id != value.target_id
            or rebuilt.minimum_angle != value.minimum_angle
            or rebuilt.hard != value.hard
        ):
            raise ValueError("Stored mesh-quality target differs from its fields.")
        return rebuilt


def _stored_tuple(value: object, name: str, /) -> tuple[Any, ...]:
    if not isinstance(value, tuple) or type(value) is not tuple:
        raise TypeError(f"Stored {name} must be a tuple.")
    return value


def _restore_size_controls(
    values: object, /, *, source_model: BRepModel | None = None
) -> tuple[SizeControl, ...]:
    stored = _stored_tuple(values, "size_controls")
    rebuilt: list[SizeControl] = []
    for control in stored:
        if type(control) is UniformSizeControl:
            rebuilt.append(
                UniformSizeControl.from_stored_fields(control, source_model=source_model)
            )
        elif type(control) is CurvatureSizeControl:
            rebuilt.append(
                CurvatureSizeControl.from_stored_fields(
                    control, source_model=source_model
                )
            )
        elif type(control) is ProximitySizeControl:
            rebuilt.append(
                ProximitySizeControl.from_stored_fields(
                    control, source_model=source_model
                )
            )
        else:
            raise TypeError("Stored size controls contain an unsupported exact type.")
    return tuple(rebuilt)


def _restore_planar_embedding(value: PlanarEmbedding | None, /) -> PlanarEmbedding | None:
    if value is None:
        return None
    if type(value) is not PlanarEmbedding:
        raise TypeError("Stored planar embedding must be PlanarEmbedding or None.")
    rebuilt = PlanarEmbedding(value.origin, value.x_axis, value.y_axis, value.normal)
    if (
        rebuilt.embedding_id != value.embedding_id
        or rebuilt.origin != value.origin
        or rebuilt.x_axis != value.x_axis
        or rebuilt.y_axis != value.y_axis
        or rebuilt.normal != value.normal
    ):
        raise ValueError("Stored planar embedding differs from its canonical frame.")
    return rebuilt


class SurfaceMeshingSpec(StrictModule, NonTrainableState):
    target: CellMeshingTarget
    planar_embedding: PlanarEmbedding | None = eqx.field(static=True)
    scope: MeshingScope
    size_controls: tuple[SizeControl, ...]
    protected_features: tuple[ProtectedFeature, ...]
    region_controls: tuple[RegionControl, ...]
    patch_controls: tuple[PatchControl, ...]
    periodic_constraints: tuple[PeriodicConstraint, ...]
    layer_controls: tuple[BoundaryLayerControl, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: SizeCompliancePolicy
    maximum_deviation: float | None = eqx.field(static=True)
    maximum_normal_deviation: float | None = eqx.field(static=True)
    quality_target: MeshQualityTarget | None
    cross_field: SurfaceCrossField | None
    background_metric: BackgroundMetricControl | None
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
        layer_controls: tuple[BoundaryLayerControl, ...] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: SizeCompliancePolicy | None = None,
        maximum_deviation: float | None = None,
        maximum_normal_deviation: float | None = None,
        quality_target: MeshQualityTarget | None = None,
        background_metric: BackgroundMetricControl | None = None,
        cross_field: SurfaceCrossField | None = None,
        deterministic: bool = True,
    ) -> None:
        if not isinstance(target, CellMeshingTarget) or target.topological_dimension != 2:
            raise ValueError(
                "Surface meshing target must have topological dimension two."
            )
        if planar_embedding is not None and not isinstance(
            planar_embedding, PlanarEmbedding
        ):
            raise TypeError("planar_embedding must be a PlanarEmbedding or None.")
        if planar_embedding is not None and target.ambient_dimension != 2:
            raise ValueError(
                "PlanarEmbedding applies only to an intrinsic two-dimensional output."
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
        layers = tuple(layer_controls)
        if not all(isinstance(control, BoundaryLayerControl) for control in layers):
            raise TypeError(
                "layer_controls must contain only BoundaryLayerControl values."
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
                "Surface boundary layers grow from wall curves of the meshed source."
            )
        deviation = _source_deviation(maximum_deviation)
        normal_deviation = (
            None
            if maximum_normal_deviation is None
            else finite_real_scalar(maximum_normal_deviation, "maximum_normal_deviation")
        )
        if normal_deviation is not None and not 0.0 <= normal_deviation <= np.pi:
            raise ValueError("maximum_normal_deviation must lie in [0, pi] radians.")
        if quality_target is not None and not isinstance(
            quality_target, MeshQualityTarget
        ):
            raise TypeError("quality_target must be MeshQualityTarget or None.")
        if background_metric is not None:
            if not isinstance(background_metric, BackgroundMetricControl):
                raise TypeError(
                    "background_metric must be BackgroundMetricControl or None."
                )
            if (
                background_metric.mesh.ambient_dimension != target.ambient_dimension
                or background_metric.metric.values.shape[1:]
                != (target.ambient_dimension, target.ambient_dimension)
            ):
                raise ValueError(
                    "The background metric must use the target ambient space."
                )
        if cross_field is not None:
            if not isinstance(cross_field, SurfaceCrossField):
                raise TypeError("cross_field must be SurfaceCrossField or None.")
            if cross_field.source.ambient_dimension != target.ambient_dimension:
                raise ValueError("The cross field must use the target ambient space.")
            if cross_field.field_id != cross_field._scientific_id():
                raise ValueError("The cross field does not match its scientific identity.")
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
        self.maximum_deviation = deviation
        self.maximum_normal_deviation = normal_deviation
        self.quality_target = quality_target
        self.background_metric = background_metric
        self.cross_field = cross_field
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
                "maximum_deviation": deviation,
                "maximum_normal_deviation": normal_deviation,
                "quality_target": None
                if quality_target is None
                else quality_target.target_id,
                "background_metric": (
                    None if background_metric is None else background_metric.control_id
                ),
                "cross_field": None if cross_field is None else cross_field.field_id,
                "deterministic": bool(deterministic),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SurfaceMeshingSpec],
        value: SurfaceMeshingSpec,
        /,
        *,
        source_model: BRepModel | None = None,
    ) -> SurfaceMeshingSpec:
        if type(value) is not cls:
            raise TypeError("value must be SurfaceMeshingSpec.")
        if type(value.deterministic) is not bool:
            raise TypeError("Stored SurfaceMeshingSpec deterministic must be bool.")
        target = CellMeshingTarget.from_stored_fields(value.target)
        planar_embedding = _restore_planar_embedding(value.planar_embedding)
        scope = MeshingScope.from_stored_fields(value.scope, source_model=source_model)
        size_controls = _restore_size_controls(
            value.size_controls, source_model=source_model
        )
        protected_features = tuple(
            ProtectedFeature.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.protected_features, "protected_features")
        )
        region_controls = tuple(
            RegionControl.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.region_controls, "region_controls")
        )
        patch_controls = tuple(
            PatchControl.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.patch_controls, "patch_controls")
        )
        periodic_constraints = tuple(
            PeriodicConstraint.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.periodic_constraints, "periodic_constraints")
        )
        layer_controls = tuple(
            BoundaryLayerControl.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.layer_controls, "layer_controls")
        )
        size_compliance = SizeCompliancePolicy.from_stored_fields(value.size_compliance)
        quality_target = (
            None
            if value.quality_target is None
            else MeshQualityTarget.from_stored_fields(value.quality_target)
        )
        background_metric = (
            None
            if value.background_metric is None
            else BackgroundMetricControl.from_stored_fields(
                value.background_metric, source_model=source_model
            )
        )
        cross_field = (
            None
            if value.cross_field is None
            else SurfaceCrossField.from_stored_fields(value.cross_field)
        )
        rebuilt = cls(
            target,
            scope,
            planar_embedding=planar_embedding,
            size_controls=size_controls,
            protected_features=protected_features,
            region_controls=region_controls,
            patch_controls=patch_controls,
            periodic_constraints=periodic_constraints,
            layer_controls=layer_controls,
            size_combination=value.size_combination,
            size_compliance=size_compliance,
            maximum_deviation=value.maximum_deviation,
            maximum_normal_deviation=value.maximum_normal_deviation,
            quality_target=quality_target,
            background_metric=background_metric,
            cross_field=cross_field,
            deterministic=value.deterministic,
        )
        if (
            rebuilt.specification_id != value.specification_id
            or rebuilt.target.target_id != value.target.target_id
            or (
                None
                if rebuilt.planar_embedding is None
                else rebuilt.planar_embedding.embedding_id
            )
            != (
                None
                if value.planar_embedding is None
                else value.planar_embedding.embedding_id
            )
            or rebuilt.scope.scope_id != value.scope.scope_id
            or tuple(control.control_id for control in rebuilt.size_controls)
            != tuple(control.control_id for control in value.size_controls)
            or rebuilt.size_combination is not value.size_combination
            or rebuilt.size_compliance.policy_id != value.size_compliance.policy_id
            or tuple(feature.feature_id for feature in rebuilt.protected_features)
            != tuple(feature.feature_id for feature in value.protected_features)
            or tuple(control.control_id for control in rebuilt.region_controls)
            != tuple(control.control_id for control in value.region_controls)
            or tuple(control.control_id for control in rebuilt.patch_controls)
            != tuple(control.control_id for control in value.patch_controls)
            or tuple(
                constraint.constraint_id for constraint in rebuilt.periodic_constraints
            )
            != tuple(
                constraint.constraint_id for constraint in value.periodic_constraints
            )
            or tuple(control.control_id for control in rebuilt.layer_controls)
            != tuple(control.control_id for control in value.layer_controls)
            or rebuilt.maximum_deviation != value.maximum_deviation
            or rebuilt.maximum_normal_deviation != value.maximum_normal_deviation
            or (
                None
                if rebuilt.quality_target is None
                else rebuilt.quality_target.target_id
            )
            != (None if value.quality_target is None else value.quality_target.target_id)
            or (
                None
                if rebuilt.background_metric is None
                else rebuilt.background_metric.control_id
            )
            != (
                None
                if value.background_metric is None
                else value.background_metric.control_id
            )
            or (
                None if rebuilt.cross_field is None else rebuilt.cross_field.field_id
            )
            != (None if value.cross_field is None else value.cross_field.field_id)
            or rebuilt.deterministic is not value.deterministic
        ):
            raise ValueError(
                "Stored SurfaceMeshingSpec differs from its canonical fields."
            )
        return rebuilt


class SurfaceRemeshingSpec(StrictModule, NonTrainableState):
    """Surface remeshing bound to one existing source mesh identity."""

    surface: SurfaceMeshingSpec
    source_mesh_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    @checked
    def __init__(self, surface: SurfaceMeshingSpec, source_mesh_id: str, /) -> None:
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


def _require_source_layer_scopes(
    control: BoundaryLayerControl, boundary_scope: MeshingScope, /
) -> None:
    """Bind one layer control to the volume request's source in source terms.

    A face ``boundary_scope`` (the source boundary) must contain the wall and cap
    faces, and the volume scope names solids of the same source binding. A solid
    ``boundary_scope`` must contain the volume scope.
    """
    binding = (
        boundary_scope.source_id,
        boundary_scope.source_revision,
        boundary_scope.entity_kind,
    )
    volume = control.volume_scope
    scopes = tuple(
        scope
        for scope in (control.wall_scope, control.cap_scope, volume)
        if scope is not None
    )
    if any(
        (scope.source_id, scope.source_revision, scope.entity_kind) != binding
        for scope in scopes
    ):
        raise ValueError(
            "Boundary-layer scopes must bind the volume request's source in "
            "source-geometry terms."
        )
    if volume is None or volume.entity_dimension != 3:
        raise ValueError(
            "Boundary-layer controls of a volume request need a solid scope."
        )
    match boundary_scope.entity_dimension:
        case 2:
            faces = tuple(
                scope
                for scope in (control.wall_scope, control.cap_scope)
                if scope is not None
            )
            if any(
                scope.entity_dimension != 2
                or scope.entity_set_id != boundary_scope.entity_set_id
                or np.setdiff1d(
                    np.asarray(scope.entity_ids), np.asarray(boundary_scope.entity_ids)
                ).size
                for scope in faces
            ):
                raise ValueError(
                    "Boundary-layer wall and cap faces must be contained in the "
                    "source boundary scope."
                )
        case 3:
            if (
                volume.entity_set_id != boundary_scope.entity_set_id
                or np.setdiff1d(
                    np.asarray(volume.entity_ids), np.asarray(boundary_scope.entity_ids)
                ).size
            ):
                raise ValueError(
                    "Boundary-layer volume scopes must be contained in the top-level "
                    "volume scope."
                )
        case _:
            raise ValueError(
                "Boundary-layer controls require a face or solid boundary scope."
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
    layer_controls: tuple[BoundaryLayerControl, ...]
    periodic_constraints: tuple[PeriodicConstraint, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: SizeCompliancePolicy
    maximum_deviation: float | None = eqx.field(static=True)
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
        layer_controls: tuple[BoundaryLayerControl, ...] = (),
        periodic_constraints: tuple[PeriodicConstraint, ...] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: SizeCompliancePolicy | None = None,
        maximum_deviation: float | None = None,
        deterministic: bool = True,
    ) -> None:
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
        if not all(
            isinstance(control, BoundaryLayerControl) for control in layer_controls_
        ):
            raise TypeError(
                "layer_controls must contain only BoundaryLayerControl values."
            )
        for control in layer_controls_:
            _require_source_layer_scopes(control, boundary_scope)
        binding = (
            boundary_scope.source_id,
            boundary_scope.source_revision,
            boundary_scope.entity_kind,
        )
        if any(
            (seed.scope.source_id, seed.scope.source_revision, seed.scope.entity_kind)
            != binding
            for seed in hole_seeds_
        ):
            raise ValueError("Volume meshing controls must share one source binding.")
        deviation = _source_deviation(maximum_deviation)
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
        self.maximum_deviation = deviation
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
                "maximum_deviation": deviation,
                "deterministic": bool(deterministic),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[VolumeMeshingSpec],
        value: VolumeMeshingSpec,
        /,
        *,
        source_model: BRepModel | None = None,
    ) -> VolumeMeshingSpec:
        if type(value) is not cls:
            raise TypeError("value must be VolumeMeshingSpec.")
        if type(value.deterministic) is not bool:
            raise TypeError("Stored VolumeMeshingSpec deterministic must be bool.")
        target = CellMeshingTarget.from_stored_fields(value.target)
        boundary_scope = MeshingScope.from_stored_fields(
            value.boundary_scope, source_model=source_model
        )
        size_controls = _restore_size_controls(
            value.size_controls, source_model=source_model
        )
        protected_features = tuple(
            ProtectedFeature.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.protected_features, "protected_features")
        )
        region_controls = tuple(
            RegionControl.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.region_controls, "region_controls")
        )
        patch_controls = tuple(
            PatchControl.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.patch_controls, "patch_controls")
        )
        region_seeds = tuple(
            RegionSeed.from_stored_fields(item)
            for item in _stored_tuple(value.region_seeds, "region_seeds")
        )
        hole_seeds = tuple(
            HoleSeed.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.hole_seeds, "hole_seeds")
        )
        layer_controls = tuple(
            BoundaryLayerControl.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.layer_controls, "layer_controls")
        )
        periodic_constraints = tuple(
            PeriodicConstraint.from_stored_fields(item, source_model=source_model)
            for item in _stored_tuple(value.periodic_constraints, "periodic_constraints")
        )
        size_compliance = SizeCompliancePolicy.from_stored_fields(value.size_compliance)
        rebuilt = cls(
            target,
            boundary_scope,
            value.fill_strategy,
            size_controls=size_controls,
            protected_features=protected_features,
            region_controls=region_controls,
            patch_controls=patch_controls,
            region_seeds=region_seeds,
            hole_seeds=hole_seeds,
            layer_controls=layer_controls,
            periodic_constraints=periodic_constraints,
            size_combination=value.size_combination,
            size_compliance=size_compliance,
            maximum_deviation=value.maximum_deviation,
            deterministic=value.deterministic,
        )
        if (
            rebuilt.specification_id != value.specification_id
            or rebuilt.target.target_id != value.target.target_id
            or rebuilt.boundary_scope.scope_id != value.boundary_scope.scope_id
            or rebuilt.fill_strategy is not value.fill_strategy
            or tuple(control.control_id for control in rebuilt.size_controls)
            != tuple(control.control_id for control in value.size_controls)
            or rebuilt.size_combination is not value.size_combination
            or rebuilt.size_compliance.policy_id != value.size_compliance.policy_id
            or tuple(feature.feature_id for feature in rebuilt.protected_features)
            != tuple(feature.feature_id for feature in value.protected_features)
            or tuple(control.control_id for control in rebuilt.region_controls)
            != tuple(control.control_id for control in value.region_controls)
            or tuple(control.control_id for control in rebuilt.patch_controls)
            != tuple(control.control_id for control in value.patch_controls)
            or tuple(seed.seed_id for seed in rebuilt.region_seeds)
            != tuple(seed.seed_id for seed in value.region_seeds)
            or tuple(seed.seed_id for seed in rebuilt.hole_seeds)
            != tuple(seed.seed_id for seed in value.hole_seeds)
            or tuple(control.control_id for control in rebuilt.layer_controls)
            != tuple(control.control_id for control in value.layer_controls)
            or tuple(
                constraint.constraint_id for constraint in rebuilt.periodic_constraints
            )
            != tuple(
                constraint.constraint_id for constraint in value.periodic_constraints
            )
            or rebuilt.maximum_deviation != value.maximum_deviation
            or rebuilt.deterministic is not value.deterministic
        ):
            raise ValueError(
                "Stored VolumeMeshingSpec differs from its canonical fields."
            )
        return rebuilt


CurveEnd: TypeAlias = Literal["start", "end"]


class CurveJunction(StrictModule, NonTrainableState):
    """Declared source endpoints sharing one vertex, never a proximity weld."""

    name: str = eqx.field(static=True)
    endpoints: tuple[tuple[int, CurveEnd], ...] = eqx.field(static=True)
    junction_id: str = eqx.field(static=True)

    def __init__(self, name: str, endpoints: tuple[tuple[int, CurveEnd], ...], /) -> None:
        name_ = normalized_identifier(name, "name")
        entries = tuple(endpoints)
        if len(entries) < 2:
            raise ValueError("A curve junction must join at least two endpoints.")
        normalized: list[tuple[int, CurveEnd]] = []
        for curve, end in entries:
            normalized.append(
                (nonnegative_integer(curve, "curve"), parse(end, CurveEnd, "end"))
            )
        ordered = tuple(sorted(normalized))
        if len(set(ordered)) != len(ordered):
            raise ValueError("Curve junction endpoints must be unique.")
        self.name = name_
        self.endpoints = ordered
        self.junction_id = canonical_fingerprint(
            {"kind": "curve-junction", "name": name_, "endpoints": ordered}
        )


class CurveMeshingSpec(StrictModule, NonTrainableState):
    """Physical interval sizing and source accuracy for an explicit curve network."""

    target: CellMeshingTarget
    scope: MeshingScope
    size_controls: tuple[SizeControl, ...]
    protected_features: tuple[ProtectedFeature, ...]
    junctions: tuple[CurveJunction, ...]
    size_combination: SizeCombinationPolicy = eqx.field(static=True)
    size_compliance: SizeCompliancePolicy
    maximum_deviation: float | None = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        target: CellMeshingTarget,
        scope: MeshingScope,
        /,
        *,
        size_controls: tuple[SizeControl, ...],
        protected_features: tuple[ProtectedFeature, ...] = (),
        junctions: tuple[CurveJunction, ...] = (),
        size_combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
        size_compliance: SizeCompliancePolicy | None = None,
        maximum_deviation: float | None = None,
        deterministic: bool = True,
    ) -> None:
        if target.topological_dimension != 1:
            raise ValueError("Curve meshing requires a dimension-one target.")
        families = target.cell_families
        selected = set(
            (*families.required, *families.preferred, *families.allowed_transitions)
        )
        if selected != {"interval"}:
            raise ValueError("Curve meshing requires only interval cells.")
        sizes, features, _, _, _, compliance = _validated_semantic_controls(
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
        if any(feature.feature_kind is FeatureKind.SURFACE for feature in features):
            raise ValueError("Curve meshing cannot protect a surface feature.")
        junctions_ = tuple(junctions)
        if not all(isinstance(junction, CurveJunction) for junction in junctions_):
            raise TypeError("junctions must contain CurveJunction values.")
        names = tuple(junction.name for junction in junctions_)
        if len(set(names)) != len(names):
            raise ValueError("Curve junction names must be unique.")
        endpoints = tuple(
            endpoint for junction in junctions_ for endpoint in junction.endpoints
        )
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("A curve endpoint can belong to only one junction.")
        curves = np.asarray(tuple(curve for curve, _ in endpoints), dtype=np.int64)
        if not np.all(np.isin(curves, np.asarray(scope.entity_ids))):
            raise ValueError("Junction curves must belong to the selected source scope.")
        ordered = tuple(sorted(junctions_, key=lambda junction: junction.endpoints))
        deviation = _source_deviation(maximum_deviation)
        self.target = target
        self.scope = scope
        self.size_controls = sizes
        self.protected_features = features
        self.junctions = ordered
        self.size_combination = size_combination
        self.size_compliance = compliance
        self.maximum_deviation = deviation
        self.deterministic = bool(deterministic)
        self.specification_id = canonical_fingerprint(
            {
                "kind": "curve-meshing-spec",
                "target": target.target_id,
                "scope": scope.scope_id,
                "size_controls": [control.control_id for control in sizes],
                "size_combination": size_combination.value,
                "size_compliance": compliance.policy_id,
                "protected_features": [feature.feature_id for feature in features],
                "junctions": [junction.junction_id for junction in ordered],
                "maximum_deviation": deviation,
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
    ) -> None:
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
        evidence: MeshingFailureEvidence | None = None,
    ) -> None:
        if not isinstance(category, MeshingFailureCategory):
            raise TypeError("category must be MeshingFailureCategory.")
        text = str(message).strip()
        if not text:
            raise ValueError("Meshing failures require a message.")
        if evidence is not None and not isinstance(evidence, MeshingFailureEvidence):
            raise TypeError("evidence must be MeshingFailureEvidence or None.")
        self.category = category
        self.provider_code = str(provider_code)
        self.stage = str(stage)
        self.entity_ids = tuple(entity_ids)
        self.locations = tuple(
            tuple(float(component) for component in point) for point in locations
        )
        self.evidence = MeshingFailureEvidence() if evidence is None else evidence
        self.evidence_id = self.evidence.evidence_id
        super().__init__(text)


__all__ = [
    "CellFamilyPolicy",
    "CellMeshingTarget",
    "CurveEnd",
    "CurveJunction",
    "CurveMeshingSpec",
    "MeshQualityTarget",
    "MeshingCapability",
    "MeshingDerivativeMode",
    "MeshingExecutionMode",
    "MeshingFailure",
    "MeshingFailureEvidence",
    "MeshingFailureCategory",
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
