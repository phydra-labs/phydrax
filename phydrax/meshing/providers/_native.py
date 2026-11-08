#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical plan/execute boundary for native initial meshing."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final, NoReturn, overload, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import numpy as np

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import canonical_fingerprint
from ..._meshcore import meshcore_identity
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._compartments import CompartmentMeshingSource
from ...geometry._meshing_domain import MeshingDomain
from ...geometry.brep._model import BRepModel
from ...geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ...geometry.multiregion_surface._label_extraction import (
    LabelFieldSurfaceExtractionResult,
)
from ...geometry.reconstruction._core import ReconstructedGeometrySource
from ...geometry.surface._model import SurfaceModel
from ...graph import GraphPartitionPlan
from ...typing import checked
from .._canonical import _accept_native_construction, _NATIVE_PROVIDER
from .._certification import _ComputedSurfaceAcceptance
from .._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveMeshingSpec,
    MeshingFailure,
    MeshingFailureCategory,
    MeshQualityTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import BoundaryLayerRoute
from .._native_construction import NativeMeshConstruction
from .._result import CellMeshingResult
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._surface_envelope import SurfaceEnvelopeSource
from ._native_options import NativeMeshingOptions, NativeMeshingRoute
from ._native_periodic import NativePeriodicSource, PreparedNativePeriodic
from ._native_source_type import NativeMeshingSource
from ._native_sources import (
    NativeCurveSource,
    NativeImplicitSource,
    NativeLayerCoreSource,
    NativeMappedHexSource,
    NativePlanarSource,
    NativePlcSource,
    NativePolyhedralSource,
    NativeStructuredSource,
    NativeSweepSource,
)


if TYPE_CHECKING:
    from .._assembly import MeshAssembly


_Specification = CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec


@runtime_checkable
class _PreparedConstruction(Protocol):
    def execute(self, /) -> NativeMeshConstruction: ...


def _failure(
    category: MeshingFailureCategory,
    message: str,
    /,
    *,
    stage: str = "source_inspection",
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code="phydrax-native",
        stage=stage,
    )


def _unsupported(
    route: NativeMeshingRoute,
    source: object,
    specification: object,
    reason: str,
    /,
) -> NoReturn:
    source_name = f"{type(source).__module__}.{type(source).__qualname__}"
    specification_name = (
        f"{type(specification).__module__}.{type(specification).__qualname__}"
    )
    raise _failure(
        MeshingFailureCategory.UNSUPPORTED_COMBINATION,
        f"Native route {route!r} does not support {source_name} with "
        f"{specification_name}: {reason}",
    )


def _source_ids(source: NativeMeshingSource, /) -> tuple[str, str] | None:
    if isinstance(source, BRepModel):
        return source.source_id, source.source_revision
    if isinstance(source, MeshingDomain):
        return source.source_id, source.source_revision
    if isinstance(source, SurfaceModel):
        return source.metadata.source_id, source.metadata.source_revision
    if isinstance(source, (AnalyticImplicitProfile, ReconstructedGeometrySource)):
        return source.source_id, source.source_revision
    if isinstance(source, CompartmentMeshingSource):
        return source.source_id, source.source_revision
    if isinstance(source, LabelFieldSurfaceExtractionResult):
        return None
    if isinstance(source, SurfaceEnvelopeSource):
        return source.source_id, source.source_revision
    if isinstance(
        source,
        (
            NativeCurveSource,
            NativeImplicitSource,
            NativeLayerCoreSource,
            NativeMappedHexSource,
            NativePlanarSource,
            NativePeriodicSource,
            NativePlcSource,
            NativePolyhedralSource,
            NativeStructuredSource,
            NativeSweepSource,
        ),
    ):
        return source.source_id, source.source_revision
    raise TypeError("source must be a NativeMeshingSource.")


def _source_identity(source: NativeMeshingSource, /) -> tuple[object, ...]:
    kind = f"{type(source).__module__}.{type(source).__qualname__}"
    if isinstance(source, BRepModel):
        return kind, source.source_id, source.source_revision, source.model_id
    if isinstance(source, MeshingDomain):
        return (
            kind,
            source.source_id,
            source.source_revision,
            source.domain_id,
            source.authority_id,
        )
    if isinstance(source, SurfaceModel):
        return (
            kind,
            source.metadata.source_id,
            source.metadata.source_revision,
            source.model_id,
            source.mesh.mesh_id,
        )
    if isinstance(source, AnalyticImplicitProfile):
        return kind, source.source_id, source.source_revision, source.profile_id
    if isinstance(source, ReconstructedGeometrySource):
        return kind, source.source_id, source.source_revision, source.binding_id
    if isinstance(source, CompartmentMeshingSource):
        return kind, source.source_id, source.source_revision
    if isinstance(source, LabelFieldSurfaceExtractionResult):
        return kind, source.lineage.lineage_id, source.evidence.evidence_id
    if isinstance(source, SurfaceEnvelopeSource):
        return kind, source.source_id, source.source_revision, source.envelope_id
    if isinstance(
        source,
        (
            NativeCurveSource,
            NativeImplicitSource,
            NativeLayerCoreSource,
            NativeMappedHexSource,
            NativePlanarSource,
            NativePeriodicSource,
            NativePlcSource,
            NativePolyhedralSource,
            NativeStructuredSource,
            NativeSweepSource,
        ),
    ):
        return kind, source.source_id, source.source_revision, source.binding_id
    raise TypeError("source must be a NativeMeshingSource.")


def _source_coordinate_contract(
    source: NativeMeshingSource, /
) -> SpatialCoordinateContract | None:
    if isinstance(source, NativeLayerCoreSource):
        from ...geometry._mesh_certificates import _same_tree_content

        owners = tuple(
            value.coordinate_contract
            for value in (
                source.source_wall_charts,
                source.source_boundary_charts,
                source.mapped_domain,
            )
            if value is not None
        )
        if not owners:
            return None
        for owner in owners:
            rebuilt = SpatialCoordinateContract.from_dict(owner.to_dict())
            if not _same_tree_content(owner, rebuilt) or not _same_tree_content(
                owner, owners[0]
            ):
                raise _failure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "Layer/core source coordinate owners disagree or are stale.",
                )
        return owners[0]
    if isinstance(source, BRepModel):
        return source.coordinate_contract
    if isinstance(source, MeshingDomain):
        authority = source.brep_authority
        return authority.coordinate_contract if isinstance(authority, BRepModel) else None
    if isinstance(source, AnalyticImplicitProfile):
        return source.coordinate_contract
    if isinstance(source, CompartmentMeshingSource):
        return source.coordinate_contract
    if isinstance(source, SurfaceModel):
        return source.metadata.coordinate_contract
    if isinstance(source, SurfaceEnvelopeSource):
        return source.coordinate_contract
    return None


def _resolve_coordinate_contract(
    source: NativeMeshingSource,
    supplied: SpatialCoordinateContract | None,
    /,
) -> SpatialCoordinateContract:
    if supplied is not None and not isinstance(supplied, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract or None.")
    owned = _source_coordinate_contract(source)
    if supplied is None:
        if owned is None:
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "A SpatialCoordinateContract is required because this source does "
                "not carry its coordinate frame and units.",
            )
        return owned
    if owned is not None:
        from ...geometry._mesh_certificates import _same_tree_content

        rebuilt = SpatialCoordinateContract.from_dict(supplied.to_dict())
        mismatch = not _same_tree_content(supplied, rebuilt) or not _same_tree_content(
            owned, supplied
        )
    else:
        mismatch = False
    if mismatch:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "coordinate_contract must match the source's bound units, coordinate "
            "system, and reference frame.",
        )
    return supplied


def _specification_scope(specification: _Specification, /) -> MeshingScope:
    if isinstance(specification, VolumeMeshingSpec):
        return specification.boundary_scope
    return specification.scope


def _validate_scope_binding(
    source: NativeMeshingSource, specification: _Specification, /
) -> None:
    identity = _source_ids(source)
    if identity is None:
        return
    scope = _specification_scope(specification)
    if (scope.source_id, scope.source_revision) != identity:
        raise _failure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "The meshing scope must bind the supplied source ID and revision.",
        )


def _families(specification: SurfaceMeshingSpec | VolumeMeshingSpec, /) -> set[str]:
    policy = specification.target.cell_families
    return set((*policy.required, *policy.preferred))


def _resolve_plc_route(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> NativeMeshingRoute:
    if isinstance(source, NativePolyhedralSource):
        return "plc_restricted_power"
    strategy = specification.fill_strategy
    if strategy in (
        VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION,
        VolumeFillStrategy.HEX_DOMINANT_POLYHEDRAL_TRANSITION,
    ):
        return "plc_hex_dominant"
    if strategy is VolumeFillStrategy.POLYHEDRAL:
        return "plc_restricted_power"
    if strategy in (VolumeFillStrategy.SWEEP, VolumeFillStrategy.MULTIZONE):
        _unsupported(
            "plc_tetrahedral",
            source,
            specification,
            f"fill_strategy {strategy.value!r} requires a NativeSweepSource or "
            "NativeStructuredSource, not a PLC source",
        )
    selected = _families(specification)
    if "hexahedron" in selected and options.grid_schedule is not None:
        return (
            "plc_balanced_grid_hex"
            if options.grid_schedule.route == "balanced_grid"
            else "plc_frame_grid_hex"
        )
    required = set(specification.target.cell_families.required)
    preferred = set(specification.target.cell_families.preferred)
    if "hexahedron" in required or (
        "hexahedron" in preferred and "tetrahedron" not in required
    ):
        return "plc_dual_hex"
    if "polyhedron" in required:
        _unsupported(
            "plc_tetrahedral",
            source,
            specification,
            "polyhedral output requires VolumeFillStrategy.POLYHEDRAL",
        )
    return "plc_tetrahedral"


def _resolve_route(
    source: NativeMeshingSource,
    specification: _Specification,
    options: NativeMeshingOptions,
    /,
) -> NativeMeshingRoute:
    if isinstance(source, NativeLayerCoreSource) and isinstance(
        specification, VolumeMeshingSpec
    ):
        if options.route not in (None, "layer_core"):
            _unsupported(
                options.route,
                source,
                specification,
                "an accepted layer/core source must retain its layer_core route",
            )
        return "layer_core"
    if options.route is not None:
        return options.route
    if isinstance(source, NativeCurveSource) and isinstance(
        specification, CurveMeshingSpec
    ):
        return "curve_arc_length"
    if isinstance(source, NativePlanarSource) and isinstance(
        specification, SurfaceMeshingSpec
    ):
        families = _families(specification)
        if (
            "quadrilateral" in families
            and "triangle" not in specification.target.cell_families.required
        ):
            return "planar_dual_quad"
        return "planar_constrained_delaunay"
    if isinstance(source, AnalyticImplicitProfile):
        if isinstance(specification, SurfaceMeshingSpec):
            return "implicit_surface"
        if isinstance(specification, VolumeMeshingSpec):
            from ...geometry.implicit._policy import AdaptiveImplicitSurfacePolicy

            if isinstance(options.implicit_policy, AdaptiveImplicitSurfacePolicy):
                return "implicit_adaptive_tetrahedral"
            if options.implicit_policy is not None:
                _unsupported(
                    "implicit_restricted_delaunay",
                    source,
                    specification,
                    "the established-reach restricted route does not consume an "
                    "adaptive implicit policy",
                )
            return "implicit_restricted_delaunay"
    if isinstance(source, NativeImplicitSource):
        return (
            "implicit_surface"
            if isinstance(specification, SurfaceMeshingSpec)
            else "implicit_adaptive_tetrahedral"
        )
    if isinstance(source, ReconstructedGeometrySource) and isinstance(
        specification, SurfaceMeshingSpec
    ):
        return "reconstructed_surface"
    if isinstance(source, MeshingDomain):
        if isinstance(specification, SurfaceMeshingSpec):
            return "parametric_surface"
        if isinstance(specification, VolumeMeshingSpec):
            return "cad_tetrahedral"
    if isinstance(source, BRepModel):
        if isinstance(specification, SurfaceMeshingSpec):
            return "parametric_surface"
        if isinstance(specification, VolumeMeshingSpec):
            return "cad_tetrahedral"
    if isinstance(source, NativePeriodicSource) and isinstance(
        specification, (SurfaceMeshingSpec, VolumeMeshingSpec)
    ):
        if isinstance(specification, VolumeMeshingSpec):
            from .._periodic import PeriodicPointOrbits

            if (
                isinstance(source.domain, PeriodicPointOrbits)
                and source.ambient_dimension == 3
                and specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
                and set(specification.target.cell_families.required) == {"polyhedron"}
            ):
                return "periodic_power"
        return "periodic_delaunay"
    if isinstance(source, NativeMappedHexSource) and isinstance(
        specification, VolumeMeshingSpec
    ):
        return (
            "mapped_frame_grid_hex"
            if options.grid_schedule is not None
            and options.grid_schedule.route == "frame_grid"
            else "mapped_balanced_grid_hex"
        )
    if isinstance(source, NativeStructuredSource) and isinstance(
        specification, (SurfaceMeshingSpec, VolumeMeshingSpec)
    ):
        return "structured_transfinite"
    if isinstance(source, NativeSweepSource) and isinstance(
        specification, VolumeMeshingSpec
    ):
        return "sweep"
    if isinstance(source, SurfaceEnvelopeSource) and isinstance(
        specification, VolumeMeshingSpec
    ):
        return "surface_envelope_tetrahedral"
    if isinstance(
        source, (CompartmentMeshingSource, LabelFieldSurfaceExtractionResult)
    ) and isinstance(specification, VolumeMeshingSpec):
        return "image_material_tetrahedral"
    if isinstance(source, (NativePlcSource, NativePolyhedralSource)) and isinstance(
        specification, VolumeMeshingSpec
    ):
        return _resolve_plc_route(source, specification, options)
    source_name = f"{type(source).__module__}.{type(source).__qualname__}"
    specification_name = (
        f"{type(specification).__module__}.{type(specification).__qualname__}"
    )
    raise _failure(
        MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
        f"No native route is available for {source_name} with {specification_name}; "
        "provide an authoritative supported source and compatible specification.",
    )


def _options_for_route(
    options: NativeMeshingOptions,
    route: NativeMeshingRoute,
    /,
    *,
    force: bool = False,
) -> NativeMeshingOptions:
    if not force and options.route == route:
        return options
    implicit_routes = {
        "implicit_surface",
        "implicit_adaptive_tetrahedral",
    }
    try:
        return NativeMeshingOptions(
            route,
            implicit_policy=options.implicit_policy if route in implicit_routes else None,
            curve_schedule=(
                options.curve_schedule if route == "curve_arc_length" else None
            ),
            surface_schedule=(
                options.surface_schedule if route == "parametric_surface" else None
            ),
            volume_schedule=(
                options.volume_schedule
                if route
                in {
                    "implicit_restricted_delaunay",
                    "implicit_adaptive_tetrahedral",
                    "plc_tetrahedral",
                    "cad_tetrahedral",
                    "layer_core",
                    "surface_envelope_tetrahedral",
                    "image_material_tetrahedral",
                    "plc_dual_hex",
                    "plc_hex_dominant",
                    "plc_balanced_grid_hex",
                    "plc_frame_grid_hex",
                }
                else None
            ),
            polyhedral_schedule=(
                options.polyhedral_schedule
                if route in {"plc_restricted_power", "layer_core"}
                else None
            ),
            structured_schedule=(
                options.structured_schedule if route == "structured_transfinite" else None
            ),
            hex_core_fraction=(
                options.hex_core_fraction if route == "plc_hex_dominant" else None
            ),
            grid_schedule=(
                options.grid_schedule
                if route
                in {
                    "plc_balanced_grid_hex",
                    "plc_frame_grid_hex",
                    "mapped_balanced_grid_hex",
                    "mapped_frame_grid_hex",
                }
                else None
            ),
        )
    except (TypeError, ValueError) as error:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            f"The selected native route {route!r} cannot use the supplied numerical "
            f"schedule: {error}",
        ) from error


def _is_brep_domain(domain: MeshingDomain, /) -> bool:
    return domain.brep_authority is not None and domain.source_kinds == (
        "vertex",
        "edge",
        "face",
    )


def _layer_target_support(specification: VolumeMeshingSpec, /) -> list[str]:
    """Admit the affine mixed layer/tetrahedral-core targets of a CAD volume."""
    target = specification.target
    families = target.cell_families
    issues: list[str] = []
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
    ):
        issues.append("an affine layer/core volume in ambient dimension three")
    if "tetrahedron" not in (*families.required, *families.preferred) or not (
        families.allow_mixed
    ):
        issues.append("a mixed layer target with a tetrahedral core")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append("the simplex core-fill strategy")
    return issues


def _boundary_angle_for_radius_edge(bound: float, /) -> float:
    """Return an angle whose outward sine bound enforces the face ratio."""
    from ...geometry._interval_enclosure import interval_sin

    required = Fraction(1, 2) / Fraction.from_float(bound)
    angle = math.asin(0.5 / bound)
    while True:
        _, upper = interval_sin((np.asarray(angle), np.asarray(angle)))
        if Fraction.from_float(float(upper)) >= required:
            return angle
        next_angle = math.nextafter(angle, math.inf)
        if next_angle == angle:
            raise ValueError("A volume radius-edge bound has no representable angle.")
        angle = next_angle


def _cad_boundary_specification(
    domain: MeshingDomain,
    specification: VolumeMeshingSpec,
    radius_edge_bound: float | None,
    hard_quality: bool,
    /,
) -> SurfaceMeshingSpec:
    from ._native_volume import _target_support

    if specification.layer_controls:
        target_issues = _layer_target_support(specification)
    else:
        target_issues = _target_support(specification)
    if target_issues:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(target_issues),
        )
    scope = specification.boundary_scope
    all_faces = domain.scope_entity_ids(2)
    if (
        scope.entity_dimension != 2
        or scope.source_id != domain.source_id
        or scope.source_revision != domain.source_revision
        or scope.entity_set_id != domain.scope_entity_set_id(2)
        or not np.array_equal(
            np.sort(scope.entity_ids),
            np.sort(all_faces),
        )
    ):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Native CAD volume meshing requires a boundary scope containing every "
            "source face occurrence.",
        )
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Native CAD volume meshing supports uniform size controls on "
                "source faces only.",
            )
        size_scope = control.scope
        if (
            size_scope.source_id != domain.source_id
            or size_scope.source_revision != domain.source_revision
            or size_scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or size_scope.entity_dimension != 2
            or size_scope.entity_set_id != domain.scope_entity_set_id(2)
            or np.setdiff1d(size_scope.entity_ids, all_faces).size
        ):
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Native CAD volume size controls must bind source face occurrences.",
            )
    if specification.maximum_deviation is None:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Native CAD tetrahedral filling requires an explicit maximum_deviation; "
            "no tolerance is inferred from source tessellation or import memory.",
        )
    if specification.periodic_constraints:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "The native CAD tetrahedral route does not support periodic constraints.",
        )
    if specification.layer_controls and (
        len(specification.layer_controls) != 1
        or specification.layer_controls[0].route is not BoundaryLayerRoute.ADVANCING
    ):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "The native CAD volume grows one ADVANCING layer control from source faces.",
        )
    target = CellMeshingTarget(
        2,
        3,
        CellFamilyPolicy(required=("triangle",)),
        geometry_order=specification.target.geometry_order,
        require_conforming=specification.target.require_conforming,
    )
    return SurfaceMeshingSpec(
        target,
        scope,
        size_controls=specification.size_controls,
        protected_features=specification.protected_features,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        periodic_constraints=specification.periodic_constraints,
        quality_target=(
            None
            if radius_edge_bound is None
            else MeshQualityTarget(
                minimum_angle=_boundary_angle_for_radius_edge(radius_edge_bound),
                # A soft volume schedule aims at, but does not gate, the
                # boundary angle that its radius-edge bound implies.
                hard=hard_quality,
            )
        ),
        deterministic=specification.deterministic,
    )


@final
class _PreparedDirectBRepCadVolume(StrictModule, NonTrainableState):
    """Dependency preparation: accepted parametric boundary, then CAD fill."""

    domain: MeshingDomain
    specification: VolumeMeshingSpec
    options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    boundary_specification: SurfaceMeshingSpec
    boundary_options: NativeMeshingOptions
    boundary_preparation: _PreparedConstruction
    boundary_plan_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: MeshingDomain,
        specification: VolumeMeshingSpec,
        options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        boundary_specification: SurfaceMeshingSpec,
        boundary_options: NativeMeshingOptions,
        boundary_preparation: _PreparedConstruction,
        boundary_plan_id: str,
        /,
    ) -> None:
        self.domain = domain
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.boundary_specification = boundary_specification
        self.boundary_options = boundary_options
        self.boundary_preparation = boundary_preparation
        self.boundary_plan_id = boundary_plan_id
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "native-direct-brep-cad-volume-preparation",
                "domain": domain.domain_id,
                "boundary_specification": boundary_specification.specification_id,
                "boundary_options": boundary_options.options_id,
                "boundary_plan": boundary_plan_id,
                "volume_specification": specification.specification_id,
                "volume_options": options.options_id,
                "coordinate_contract": coordinate_contract.spatial_id,
            }
        )

    def execute(self, /) -> NativeMeshConstruction:
        boundary_construction = self.boundary_preparation.execute()
        surface_proofs: list[_ComputedSurfaceAcceptance] = []
        boundary_result = _accept_native_construction(
            boundary_construction,
            self.boundary_specification,
            _NATIVE_PROVIDER,
            meshcore_identity(),
            self.boundary_plan_id,
            "parametric_surface",
            computed_surface_out=None
            if self.specification.layer_controls
            else surface_proofs,
        )
        if self.specification.layer_controls:
            from .._layer_core_brep import brep_layer_core_source
            from ._native_layer import prepare_native_layer

            try:
                surface_schedule = self.boundary_options.surface_schedule
                if surface_schedule is None:
                    raise ValueError("The original boundary surface schedule is absent.")
                layer_source = brep_layer_core_source(
                    self.domain, boundary_result, self.specification, surface_schedule
                )
            except ValueError as error:
                raise _failure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    f"The source-face layer control has no native CAD realization: {error}",
                ) from error
            return prepare_native_layer(
                layer_source,
                self.specification,
                _options_for_route(self.options, "layer_core", force=True),
                self.coordinate_contract,
            ).execute()
        from ._native_cad_volume import NativeCadVolumeSource, prepare_native_cad_volume

        cad_source = NativeCadVolumeSource(
            self.domain,
            boundary_result,
            maximum_deviation=None,
            boundary="fixed",
            _computed_surface=surface_proofs[0],
        )
        surface_proofs.clear()
        prepared_volume = prepare_native_cad_volume(
            cad_source,
            self.specification,
            self.options,
            self.coordinate_contract,
        )
        return prepared_volume.execute()


def _prepare_brep_volume(
    source: BRepModel | MeshingDomain,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    requested_options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedDirectBRepCadVolume:
    domain = MeshingDomain.from_brep(source) if isinstance(source, BRepModel) else source
    if not _is_brep_domain(domain):
        _unsupported(
            "cad_tetrahedral",
            source,
            specification,
            "the volume route requires a BRep-authoritative MeshingDomain",
        )
    schedule = options.volume_schedule
    if schedule is None:
        raise ValueError(
            "Native CAD volume route requires its numerical volume schedule."
        )
    from .._volume_generation import _shape_targets_gated

    boundary_specification = _cad_boundary_specification(
        domain,
        specification,
        None if specification.layer_controls else schedule.radius_edge_bound,
        _shape_targets_gated(schedule),
    )
    boundary_options = _options_for_route(
        requested_options, "parametric_surface", force=True
    )
    from ._native_surface import prepare_native_surface

    boundary_preparation = prepare_native_surface(
        domain,
        boundary_specification,
        boundary_options,
        coordinate_contract,
    )
    boundary_plan_id = canonical_fingerprint(
        {
            "kind": "native-meshing-plan",
            "source": _source_identity(domain),
            "specification": boundary_specification.specification_id,
            "options": boundary_options.options_id,
            "route": "parametric_surface",
            "coordinate_contract": coordinate_contract.spatial_id,
        }
    )
    return _PreparedDirectBRepCadVolume(
        domain,
        specification,
        options,
        coordinate_contract,
        boundary_specification,
        boundary_options,
        boundary_preparation,
        boundary_plan_id,
    )


def _prepare_route(
    route: NativeMeshingRoute,
    source: NativeMeshingSource,
    specification: _Specification,
    options: NativeMeshingOptions,
    requested_options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedConstruction:
    match route:
        case "curve_arc_length":
            if not isinstance(source, NativeCurveSource):
                _unsupported(
                    route, source, specification, "a NativeCurveSource is required"
                )
            if not isinstance(specification, CurveMeshingSpec):
                _unsupported(
                    route, source, specification, "a CurveMeshingSpec is required"
                )
            from ._native_curve import prepare_native_curve

            return prepare_native_curve(
                source, specification, options, coordinate_contract
            )
        case "planar_constrained_delaunay":
            if not isinstance(source, NativePlanarSource):
                _unsupported(
                    route, source, specification, "a NativePlanarSource is required"
                )
            if not isinstance(specification, SurfaceMeshingSpec):
                _unsupported(
                    route, source, specification, "a SurfaceMeshingSpec is required"
                )
            from ._native_planar import prepare_native_planar

            return prepare_native_planar(
                source, specification, options, coordinate_contract
            )
        case "planar_dual_quad":
            if not isinstance(source, NativePlanarSource):
                _unsupported(
                    route, source, specification, "a NativePlanarSource is required"
                )
            if not isinstance(specification, SurfaceMeshingSpec):
                _unsupported(
                    route, source, specification, "a SurfaceMeshingSpec is required"
                )
            from ._native_dual import prepare_native_dual_quad

            return prepare_native_dual_quad(
                source, specification, options, coordinate_contract
            )
        case "implicit_surface":
            if not isinstance(source, (NativeImplicitSource, AnalyticImplicitProfile)):
                _unsupported(
                    route, source, specification, "an implicit source is required"
                )
            if not isinstance(specification, SurfaceMeshingSpec):
                _unsupported(
                    route, source, specification, "a SurfaceMeshingSpec is required"
                )
            from ._native_implicit import prepare_native_implicit

            return prepare_native_implicit(
                source, specification, options, coordinate_contract
            )
        case "implicit_restricted_delaunay":
            if not isinstance(source, AnalyticImplicitProfile):
                _unsupported(
                    route,
                    source,
                    specification,
                    "restricted Delaunay requires an analytic implicit profile with established reach",
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_implicit import prepare_native_implicit

            return prepare_native_implicit(
                source, specification, options, coordinate_contract
            )
        case "implicit_adaptive_tetrahedral":
            if not isinstance(source, (NativeImplicitSource, AnalyticImplicitProfile)):
                _unsupported(
                    route, source, specification, "an implicit source is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_implicit import prepare_native_implicit

            return prepare_native_implicit(
                source, specification, options, coordinate_contract
            )
        case "parametric_surface":
            if not isinstance(specification, SurfaceMeshingSpec):
                _unsupported(
                    route, source, specification, "a SurfaceMeshingSpec is required"
                )
            if isinstance(source, BRepModel):
                domain = MeshingDomain.from_brep(source)
            elif isinstance(source, (MeshingDomain, ReconstructedGeometrySource)):
                domain = source
            else:
                _unsupported(
                    route,
                    source,
                    specification,
                    "a BRepModel, MeshingDomain, or ReconstructedGeometrySource is required",
                )
            from ._native_surface import prepare_native_surface

            return prepare_native_surface(
                domain, specification, options, coordinate_contract
            )
        case "reconstructed_surface":
            if not isinstance(source, ReconstructedGeometrySource):
                _unsupported(
                    route,
                    source,
                    specification,
                    "a ReconstructedGeometrySource is required",
                )
            if not isinstance(specification, SurfaceMeshingSpec):
                _unsupported(
                    route, source, specification, "a SurfaceMeshingSpec is required"
                )
            from ._native_reconstruction import prepare_native_reconstruction

            return prepare_native_reconstruction(
                source, specification, options, coordinate_contract
            )
        case "plc_tetrahedral":
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            if isinstance(source, NativePlcSource):
                from ._native_volume import prepare_native_plc_volume

                return prepare_native_plc_volume(
                    source, specification, options, coordinate_contract
                )
            if isinstance(
                source, (CompartmentMeshingSource, LabelFieldSurfaceExtractionResult)
            ):
                from ._native_compartment import prepare_native_compartment

                return prepare_native_compartment(
                    source, specification, options, coordinate_contract
                )
            _unsupported(
                route,
                source,
                specification,
                "a NativePlcSource or categorical compartment source is required",
            )
        case "cad_tetrahedral":
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            if not isinstance(source, (BRepModel, MeshingDomain)):
                _unsupported(
                    route,
                    source,
                    specification,
                    "a BRepModel or MeshingDomain is required",
                )
            return _prepare_brep_volume(
                source,
                specification,
                options,
                requested_options,
                coordinate_contract,
            )
        case "periodic_delaunay" | "periodic_power":
            if not isinstance(source, NativePeriodicSource):
                _unsupported(
                    route, source, specification, "a NativePeriodicSource is required"
                )
            if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
                _unsupported(
                    route,
                    source,
                    specification,
                    "a surface or volume specification is required",
                )
            from ._native_periodic import prepare_native_periodic

            return prepare_native_periodic(
                source, specification, options, coordinate_contract
            )
        case "plc_restricted_power":
            if not isinstance(source, (NativePlcSource, NativePolyhedralSource)):
                _unsupported(
                    route, source, specification, "a PLC or polyhedral source is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_polyhedral import prepare_native_polyhedral

            return prepare_native_polyhedral(
                source, specification, options, coordinate_contract
            )
        case "layer_core":
            if not isinstance(source, NativeLayerCoreSource):
                _unsupported(
                    route, source, specification, "a NativeLayerCoreSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_layer import prepare_native_layer

            return prepare_native_layer(
                source, specification, options, coordinate_contract
            )
        case "surface_envelope_tetrahedral":
            if not isinstance(source, SurfaceEnvelopeSource):
                _unsupported(
                    route, source, specification, "a SurfaceEnvelopeSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from .._surface_envelope import prepare_native_surface_envelope

            return prepare_native_surface_envelope(
                source, specification, options, coordinate_contract
            )
        case "structured_transfinite":
            if not isinstance(source, NativeStructuredSource):
                _unsupported(
                    route, source, specification, "a NativeStructuredSource is required"
                )
            if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
                _unsupported(
                    route,
                    source,
                    specification,
                    "a surface or volume specification is required",
                )
            from ._native_structured import prepare_native_structured

            return prepare_native_structured(
                source, specification, options, coordinate_contract
            )
        case "sweep":
            if not isinstance(source, NativeSweepSource):
                _unsupported(
                    route, source, specification, "a NativeSweepSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_structured import prepare_native_sweep

            return prepare_native_sweep(
                source, specification, options, coordinate_contract
            )
        case "image_material_tetrahedral":
            if not isinstance(
                source, (CompartmentMeshingSource, LabelFieldSurfaceExtractionResult)
            ):
                _unsupported(
                    route,
                    source,
                    specification,
                    "a categorical compartment source is required",
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_compartment import prepare_native_compartment

            return prepare_native_compartment(
                source, specification, options, coordinate_contract
            )
        case "plc_dual_hex":
            if not isinstance(source, NativePlcSource):
                _unsupported(
                    route, source, specification, "a NativePlcSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_dual import prepare_native_dual_hex

            return prepare_native_dual_hex(
                source, specification, options, coordinate_contract
            )
        case "plc_hex_dominant":
            if not isinstance(source, NativePlcSource):
                _unsupported(
                    route, source, specification, "a NativePlcSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_dual import prepare_native_hex_dominant

            return prepare_native_hex_dominant(
                source, specification, options, coordinate_contract
            )
        case "plc_balanced_grid_hex":
            if not isinstance(source, NativePlcSource):
                _unsupported(
                    route, source, specification, "a NativePlcSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_dual import prepare_native_balanced_grid_hex

            return prepare_native_balanced_grid_hex(
                source, specification, options, coordinate_contract
            )
        case "plc_frame_grid_hex":
            if not isinstance(source, NativePlcSource):
                _unsupported(
                    route, source, specification, "a NativePlcSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_dual import prepare_native_frame_grid_hex

            return prepare_native_frame_grid_hex(
                source, specification, options, coordinate_contract
            )
        case "mapped_balanced_grid_hex":
            if not isinstance(source, NativeMappedHexSource):
                _unsupported(
                    route, source, specification, "a NativeMappedHexSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_dual import prepare_native_mapped_balanced_grid_hex

            return prepare_native_mapped_balanced_grid_hex(
                source, specification, options, coordinate_contract
            )
        case "mapped_frame_grid_hex":
            if not isinstance(source, NativeMappedHexSource):
                _unsupported(
                    route, source, specification, "a NativeMappedHexSource is required"
                )
            if not isinstance(specification, VolumeMeshingSpec):
                _unsupported(
                    route, source, specification, "a VolumeMeshingSpec is required"
                )
            from ._native_dual import prepare_native_mapped_frame_grid_hex

            return prepare_native_mapped_frame_grid_hex(
                source, specification, options, coordinate_contract
            )
    _unsupported(
        route, source, specification, "the route has no native preparation implementation"
    )


@final
class NativeMeshingPlan(StrictModule, NonTrainableState):
    """Reusable, source-bound preparation for one canonical native mesh request."""

    source: NativeMeshingSource
    specification: _Specification
    options: NativeMeshingOptions
    route_options: NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    prepared: _PreparedConstruction
    route: NativeMeshingRoute = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativeMeshingSource,
        specification: _Specification,
        options: NativeMeshingOptions,
        route_options: NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        prepared: _PreparedConstruction,
        route: NativeMeshingRoute,
        plan_id: str,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.options = options
        self.route_options = route_options
        self.coordinate_contract = coordinate_contract
        self.prepared = prepared
        self.route = route
        self.plan_id = plan_id

    @overload
    def execute(self, /, *, execution_group: None = None) -> CellMeshingResult: ...

    @overload
    def execute(
        self,
        /,
        *,
        execution_group: ExecutionGroup,
        graph_partition_plan: GraphPartitionPlan | None = None,
    ) -> MeshAssembly: ...

    def execute(
        self,
        /,
        *,
        execution_group: ExecutionGroup | None = None,
        graph_partition_plan: GraphPartitionPlan | None = None,
    ) -> CellMeshingResult | MeshAssembly:
        from ._native_implicit import _PreparedFixedImplicitRoute

        if isinstance(self.prepared, _PreparedFixedImplicitRoute):
            _require_fixed_plan_current(self, check_prepared=False)
        if graph_partition_plan is not None:
            if not isinstance(graph_partition_plan, GraphPartitionPlan):
                raise TypeError(
                    "graph_partition_plan must be GraphPartitionPlan or None."
                )
            if execution_group is None:
                raise ValueError("graph_partition_plan requires an execution_group.")
        if execution_group is not None:
            from .._assembly import MeshAssembly

            if not isinstance(execution_group, ExecutionGroup):
                raise TypeError("execution_group must be ExecutionGroup or None.")
            if not isinstance(self.prepared, PreparedNativePeriodic):
                _unsupported(
                    self.route,
                    self.source,
                    self.specification,
                    "the prepared route does not support owner-local execution",
                )
            assembly = self.prepared.execute(
                execution_group=execution_group,
                graph_partition_plan=graph_partition_plan,
            )
            if not isinstance(assembly, MeshAssembly):
                raise TypeError("An owner-local native route must return MeshAssembly.")
            return assembly
        construction = self.prepared.execute()
        if not isinstance(construction, NativeMeshConstruction):
            raise TypeError("A native prepared route must return NativeMeshConstruction.")
        return _accept_native_construction(
            construction,
            self.specification,
            _NATIVE_PROVIDER,
            meshcore_identity(),
            self.plan_id,
            self.route,
        )


def _require_fixed_plan_current(
    plan: NativeMeshingPlan, /, *, check_prepared: bool = True
) -> None:
    from ...geometry._mesh_certificates import _same_tree_content
    from ._native_implicit import _PreparedFixedImplicitRoute

    prepared = plan.prepared
    if (
        plan.route != "implicit_surface"
        or not isinstance(plan.source, NativeImplicitSource)
        or not isinstance(plan.specification, SurfaceMeshingSpec)
        or not isinstance(prepared, _PreparedFixedImplicitRoute)
        or not _same_tree_content(plan.source, prepared.source)
        or not _same_tree_content(plan.specification, prepared.specification)
        or not _same_tree_content(plan.coordinate_contract, prepared.coordinate_contract)
        or not _same_tree_content(
            plan.route_options, _options_for_route(plan.options, plan.route)
        )
    ):
        raise ValueError(
            "The fixed plan no longer matches its source or request binding."
        )
    _validate_scope_binding(plan.source, plan.specification)
    if check_prepared:
        prepared.require_current()
    expected = canonical_fingerprint(
        {
            "kind": "native-meshing-plan",
            "source": _source_identity(plan.source),
            "specification": plan.specification.specification_id,
            "options": plan.options.options_id,
            "resolved_options": plan.route_options.options_id,
            "route": plan.route,
            "coordinate_contract": plan.coordinate_contract.spatial_id,
            "preparation": prepared.preparation_id,
        }
    )
    if plan.plan_id != expected:
        raise ValueError("The fixed plan identity is no longer current.")


class NativeMeshingProvider:
    """Source- and request-driven dispatcher for native initial meshing."""

    def __init__(self, options: NativeMeshingOptions | None = None, /) -> None:
        self.options = NativeMeshingOptions() if options is None else options
        if not isinstance(self.options, NativeMeshingOptions):
            raise TypeError("options must be NativeMeshingOptions or None.")

    @checked
    def plan(
        self,
        source: NativeMeshingSource,
        specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract | None = None,
        fixed_plan: NativeMeshingPlan | None = None,
    ) -> NativeMeshingPlan:
        if not isinstance(source, NativeMeshingSource):
            raise TypeError("source must be a NativeMeshingSource.")
        if not isinstance(
            specification, (CurveMeshingSpec, SurfaceMeshingSpec, VolumeMeshingSpec)
        ):
            raise TypeError(
                "specification must be CurveMeshingSpec, SurfaceMeshingSpec, "
                "or VolumeMeshingSpec."
            )
        contract = _resolve_coordinate_contract(source, coordinate_contract)
        _validate_scope_binding(source, specification)
        route = _resolve_route(source, specification, self.options)
        if (
            isinstance(specification, SurfaceMeshingSpec)
            and specification.cross_field is not None
            and route != "planar_dual_quad"
        ):
            _unsupported(
                route,
                source,
                specification,
                "source-bound cross fields require the planar_dual_quad route",
            )
        route_options = _options_for_route(self.options, route)
        if fixed_plan is None:
            prepared = _prepare_route(
                route,
                source,
                specification,
                route_options,
                self.options,
                contract,
            )
            preparation_id = (
                prepared.preparation_id
                if isinstance(prepared, _PreparedDirectBRepCadVolume)
                else None
            )
            if route == "implicit_surface":
                from ._native_implicit import _PreparedFixedImplicitRoute

                if isinstance(prepared, _PreparedFixedImplicitRoute):
                    preparation_id = prepared.preparation_id
        else:
            from ...geometry._mesh_certificates import _same_tree_content
            from ._native_implicit import (
                _PreparedFixedImplicitRoute,
                prepare_native_fixed_rebind,
            )

            if (
                not isinstance(fixed_plan, NativeMeshingPlan)
                or not isinstance(source, NativeImplicitSource)
                or not isinstance(specification, SurfaceMeshingSpec)
                or route != "implicit_surface"
                or fixed_plan.route != route
                or not isinstance(fixed_plan.prepared, _PreparedFixedImplicitRoute)
                or fixed_plan.coordinate_contract.spatial_id != contract.spatial_id
            ):
                raise ValueError(
                    "A fixed plan requires the same fixed implicit route and coordinate frame."
                )
            if not _same_tree_content(fixed_plan.options, self.options):
                raise ValueError(
                    "The fixed rebind cannot change its authored policy or options."
                )
            _require_fixed_plan_current(fixed_plan)
            prepared = prepare_native_fixed_rebind(
                source, specification, route_options, contract, fixed_plan.prepared
            )
            preparation_id = prepared.preparation_id
        plan_id = canonical_fingerprint(
            {
                "kind": "native-meshing-plan",
                "source": _source_identity(source),
                "specification": specification.specification_id,
                "options": self.options.options_id,
                "resolved_options": route_options.options_id,
                "route": route,
                "coordinate_contract": contract.spatial_id,
                "preparation": preparation_id,
            }
        )
        return NativeMeshingPlan(
            source,
            specification,
            self.options,
            route_options,
            contract,
            prepared,
            route,
            plan_id,
        )


__all__ = ["NativeMeshingPlan", "NativeMeshingProvider"]
