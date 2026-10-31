#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare and execute the fixed-layer/native-PLC construction route."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from typing import final, NamedTuple, TYPE_CHECKING

import jax
import numpy as np

import phydrax.discretization._nested_reference as nested_reference_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.meshing._canonical as canonical_owner
import phydrax.meshing._curving as high_order_curving_owner
import phydrax.meshing._layer_core as layer_core_owner
import phydrax.meshing._layer_core_brep as brep_owner
import phydrax.meshing._layer_core_controls as controls_owner
import phydrax.meshing._layer_core_mapped as mapped_core_owner
import phydrax.meshing._layer_curving as layer_curving_owner
import phydrax.meshing._native_construction as construction_owner
import phydrax.meshing._result as result_owner
import phydrax.meshing._sizing as sizing_owner
import phydrax.meshing.providers._native_options as options_owner
import phydrax.meshing.providers._native_sources as sources_owner

from ..._meshcore import MeshcoreError, MeshcoreStatus, zero_set_remesh
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from ...discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._hexahedral import HexahedralConnectivity
from ...geometry._domain_boundary import (
    _patch_chart_polygon,
    MeshingDomainBoundarySource,
)
from ...geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._meshing_domain import MeshingDomain
from ...geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryQuery,
)
from ...geometry._source_identity import source_entity_id
from ...typing import checked
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    MeshingFailureCategory,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import (
    FeatureKind,
    PeriodicConstraint,
    ProtectedFeature,
    RegionControl,
)
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from ._native_polyhedral import polyhedral_support_issues


if TYPE_CHECKING:
    from ...geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
    )


def _lower_polyhedral_periodic_constraints(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
) -> tuple[PeriodicConstraint, ...]:
    """Rebind authored source seam scopes to the exact fixed core PLC banks."""
    core_power_source = source.core_power_source
    if core_power_source is None:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Polyhedral layer/core fill requires an authored core power source.",
            provider_code="native_layer_core_polyhedral",
            stage="route_admission",
        )
    facet_scope_ids = controls_owner.core_facet_scope_ids(source)

    def lower(scope: MeshingScope, /) -> MeshingScope:
        if scope.entity_dimension != 2:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Polyhedral layer/core periodic controls require source facet scopes.",
                provider_code="native_layer_core_polyhedral",
                stage="route_admission",
            )
        selected = np.asarray(scope.entity_ids, dtype=np.int64)
        rows = np.flatnonzero(np.isin(facet_scope_ids, selected))
        represented = set(facet_scope_ids[rows].tolist())
        if represented != set(selected.tolist()):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A periodic source facet scope is not retained by the fixed core PLC.",
                provider_code="native_layer_core_polyhedral",
                stage="route_admission",
                entity_ids=tuple(int(value) for value in selected),
            )
        identifiers = np.asarray(source.complex.facet_entity_ids, dtype=np.int64)[rows]
        return MeshingScope(
            core_power_source.source_id,
            core_power_source.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            core_power_source.scope_entity_set_id(2),
            identifiers,
        )

    return tuple(
        PeriodicConstraint(
            lower(control.source_scope),
            lower(control.target_scope),
            control.transform,
            tolerance=control.tolerance,
            conforming_required=control.conforming_required,
        )
        for control in specification.periodic_constraints
    )


def _lower_core_specification(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
) -> VolumeMeshingSpec:
    complex_ = source.complex
    polyhedral = specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
    core_power_source = source.core_power_source
    if polyhedral and core_power_source is None:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Polyhedral layer/core fill requires an authored core power source.",
            provider_code="native_layer_core_polyhedral",
            stage="route_admission",
        )
    source_boundary = source.source_boundary_scope
    core_entity_kind = (
        MeshingEntityKind.GEOMETRY
        if polyhedral
        else specification.boundary_scope.entity_kind
    )
    core_scope = MeshingScope(
        specification.boundary_scope.source_id,
        specification.boundary_scope.source_revision,
        core_entity_kind,
        2,
        core_power_source.scope_entity_set_id(2)
        if core_power_source is not None
        else f"{complex_.complex_id}:facets",
        np.asarray(complex_.facet_entity_ids, dtype=np.int64),
    )
    size = specification.size_controls[0]
    if not isinstance(size, sizing_owner.UniformSizeControl):
        raise TypeError("Layer/core filling requires one uniform physical size control.")
    if source.reference_source is None:
        size_scale = 1.0
    else:
        mapped_domain = source.mapped_domain
        if not isinstance(mapped_domain, MappedReferenceDomain):
            raise ValueError("Mapped layer core source lost its original mapped domain.")
        size_scale = mapped_core_owner.mapped_source_lipschitz_bound(mapped_domain)
    lowered_size = sizing_owner.UniformSizeControl(
        core_scope,
        size.target_size / size_scale,
        minimum_size=None
        if size.minimum_size is None
        else size.minimum_size / size_scale,
        maximum_size=None
        if size.maximum_size is None
        else size.maximum_size / size_scale,
        maximum_growth_rate=size.maximum_growth_rate,
        strength=size.strength,
        priority=size.priority,
    )
    core_regions_list: list[RegionControl] = []
    for control in specification.region_controls:
        core_row = _core_region_row(source, specification, control)
        if core_row is None:
            raise ValueError(
                "Region control identity is not retained by the fixed PLC core."
            )
        region_name = complex_.region_ids[core_row]
        core_regions_list.append(
            RegionControl(
                MeshingScope(
                    specification.boundary_scope.source_id,
                    specification.boundary_scope.source_revision,
                    core_entity_kind,
                    3,
                    core_power_source.scope_entity_set_id(3)
                    if core_power_source is not None
                    else f"{complex_.complex_id}:regions",
                    np.asarray([complex_.region_entity_ids[core_row]], dtype=np.int64),
                ),
                region_name,
                control.material_id,
                control.role,
                meshing_enabled=control.meshing_enabled,
            )
        )
    core_regions = tuple(core_regions_list)
    facet_scope_ids = controls_owner.core_facet_scope_ids(source)
    features: list[ProtectedFeature] = []
    for feature in specification.protected_features:
        if feature.scope.entity_dimension != 2:
            if source_boundary is not None:
                raise ValueError(
                    "Layer/core filling requires explicit source vertex/edge ancestry for lower-dimensional protected features."
                )
            if feature.scope.entity_dimension == 0:
                selected = np.asarray(feature.scope.entity_ids, dtype=np.int64)
                entity_set_id = (
                    core_power_source.scope_entity_set_id(0)
                    if core_power_source is not None
                    else f"{complex_.complex_id}:vertices"
                )
            elif feature.scope.entity_dimension == 1:
                selected = np.asarray(feature.scope.entity_ids, dtype=np.int64)
                entity_set_id = (
                    core_power_source.scope_entity_set_id(1)
                    if core_power_source is not None
                    else f"{complex_.complex_id}:segments"
                )
            else:
                raise ValueError("Unsupported layer/core protected-feature dimension.")
            lowered_scope = MeshingScope(
                core_scope.source_id,
                core_scope.source_revision,
                core_scope.entity_kind,
                feature.scope.entity_dimension,
                entity_set_id,
                selected,
            )
            features.append(
                ProtectedFeature(
                    lowered_scope,
                    feature.feature_kind,
                    maximum_deviation=feature.maximum_deviation,
                    hard=feature.hard,
                )
            )
            continue
        feature_ids = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        if source_boundary is None:
            selected = np.intersect1d(
                feature_ids,
                np.asarray(complex_.facet_entity_ids, dtype=np.int64),
            )
        else:
            rows = np.flatnonzero(np.isin(facet_scope_ids, feature_ids))
            selected = np.asarray(complex_.facet_entity_ids, dtype=np.int64)[rows]
        if selected.size:
            lowered_scope = MeshingScope(
                core_scope.source_id,
                core_scope.source_revision,
                core_scope.entity_kind,
                2,
                core_scope.entity_set_id,
                selected,
            )
            features.append(
                ProtectedFeature(
                    lowered_scope,
                    feature.feature_kind,
                    maximum_deviation=feature.maximum_deviation,
                    hard=feature.hard,
                )
            )
    region_names = set(complex_.region_ids)
    region_seeds = tuple(
        seed for seed in specification.region_seeds if seed.region_name in region_names
    )
    core_target = (
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("polyhedron",)),
            geometry_order=1,
            require_conforming=specification.target.require_conforming,
        )
        if polyhedral
        else specification.target
    )
    return VolumeMeshingSpec(
        core_target,
        core_scope,
        VolumeFillStrategy.POLYHEDRAL if polyhedral else VolumeFillStrategy.SIMPLEX,
        size_controls=(lowered_size,),
        protected_features=tuple(features),
        region_controls=core_regions,
        patch_controls=(),
        region_seeds=region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=(),
        periodic_constraints=(
            _lower_polyhedral_periodic_constraints(source, specification)
            if polyhedral
            else ()
        ),
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )


def _core_region_row(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    control: RegionControl,
    /,
) -> int | None:
    """Resolve a scoped authored region through its stable composite ID bank."""
    scope = control.scope
    identifiers = np.asarray(scope.entity_ids, dtype=np.int64)
    if (
        scope.source_id != source.source_id
        or scope.source_revision != source.source_revision
        or scope.entity_kind is not specification.boundary_scope.entity_kind
        or scope.entity_dimension != 3
        or scope.entity_set_id != source.scope_entity_set_id(3)
        or identifiers.shape != (1,)
    ):
        return None
    composite_rows = np.flatnonzero(source.region_entity_ids == identifiers[0])
    if composite_rows.size != 1:
        return None
    composite = int(composite_rows[0])
    core_rows = np.flatnonzero(
        np.asarray(source.core_region_map, dtype=np.int64) == composite
    )
    if core_rows.size != 1:
        return None
    core_row = int(core_rows[0])
    region_name, occurrence_path = source.source_entity_key_for_scope(
        3, int(identifiers[0])
    )
    if (
        region_name != control.region_name
        or region_name != source.complex.region_ids[core_row]
        or tuple(occurrence_path)
        != tuple(source.complex.region_occurrence_paths[core_row])
    ):
        return None
    return core_row


def _boundary_control_issues(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:
    issues: list[str] = []
    boundary = specification.boundary_scope
    if source.source_boundary_scope is None:
        chart_domain = (
            None
            if source.source_surface_charts is None
            else source.source_surface_charts.domain
        )
        retained_layer_domain = (
            chart_domain is not None or source.source_domain is not None
        )
        fixed_plc_boundary = (
            boundary.entity_dimension == 2
            and boundary.entity_set_id == source.scope_entity_set_id(2)
            and np.array_equal(
                np.sort(np.asarray(boundary.entity_ids, dtype=np.int64)),
                np.sort(np.asarray(source.complex.facet_entity_ids, dtype=np.int64)),
            )
        )
        native_volume_scope = boundary.entity_set_id == source.scope_entity_set_id(
            3
        ) and np.array_equal(
            np.asarray(boundary.entity_ids, dtype=np.int64),
            source.scope_entity_ids(3),
        )
        chart_volume_scope = (
            chart_domain is not None
            and boundary.entity_set_id == chart_domain.scope_entity_set_id(3)
            and np.array_equal(
                np.asarray(boundary.entity_ids, dtype=np.int64),
                chart_domain.scope_entity_ids(3),
            )
        )
        layer_volume_scope = (
            specification.layer_controls
            and retained_layer_domain
            and boundary.entity_dimension == 3
            and (native_volume_scope or chart_volume_scope)
        )
        if not (fixed_plc_boundary or layer_volume_scope):
            issues.append(
                "a complete fixed-core PLC facet boundary or retained layer-volume scope"
            )
        if specification.layer_controls and not retained_layer_domain:
            issues.append(
                "retained source boundary/domain facts for physical layer controls"
            )
    elif boundary.scope_id != source.source_boundary_scope.scope_id:
        issues.append("the exact retained original boundary scope")
    if len(specification.size_controls) != 1 or not isinstance(
        specification.size_controls[0], sizing_owner.UniformSizeControl
    ):
        issues.append("an explicit physical core resolution via one uniform size control")
    elif specification.size_controls[0].scope.scope_id != boundary.scope_id:
        issues.append("a whole-core size-control scope")
    return issues


def _feature_support_issues(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:
    issues: list[str] = []
    entity_ids = {
        0: np.asarray(source.complex.vertex_entity_ids, dtype=np.int64),
        1: np.asarray(source.complex.segment_entity_ids, dtype=np.int64),
        2: np.asarray(source.complex.facet_entity_ids, dtype=np.int64),
    }
    for feature in specification.protected_features:
        scope = feature.scope
        if source.source_surface_charts is None:
            query = source.fidelity_source
            if not isinstance(query, MappedReferenceBoundarySource):
                issues.append(
                    "authored source-chart ancestry for scoped protected layer features"
                )
                continue
            if (
                feature.feature_kind
                not in (FeatureKind.SURFACE, FeatureKind.MATERIAL_INTERFACE)
                or scope.entity_dimension != 2
            ):
                issues.append(
                    "typed mapped-root surface witnesses for scoped protected layer features"
                )
                continue
            if (
                scope.source_id != source.source_id
                or scope.source_revision != source.source_revision
                or scope.entity_kind is not MeshingEntityKind.GEOMETRY
                or scope.entity_set_id != source.scope_entity_set_id(2)
            ):
                issues.append(
                    "mapped-root protected feature scope bound to its original source bank"
                )
                continue
            authority_ids = np.asarray(
                query.authority.scope_entity_ids(2), dtype=np.int64
            )
            authority_keys = {
                query.authority.source_entity_key_for_scope(2, int(identifier))
                for identifier in authority_ids.tolist()
            }
            if any(
                source.source_entity_key_for_scope(2, int(identifier))
                not in authority_keys
                for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist()
            ):
                issues.append(
                    "mapped-root feature keys preserved by the original source authority"
                )
            continue
        if feature.hard and (
            feature.feature_kind
            not in (FeatureKind.SURFACE, FeatureKind.MATERIAL_INTERFACE)
            or feature.scope.entity_dimension != 2
        ):
            issues.append("typed exact source witnesses for hard non-surface features")
        dimension = scope.entity_dimension
        ids = np.asarray(scope.entity_ids, dtype=np.int64)
        if source.source_boundary_scope is not None:
            boundary = source.source_boundary_scope
            if dimension != 2:
                issues.append(
                    "explicit original vertex/edge ancestry for lower-dimensional protected features"
                )
            elif (
                scope.source_id != boundary.source_id
                or scope.source_revision != boundary.source_revision
                or scope.entity_kind is not boundary.entity_kind
                or scope.entity_set_id != boundary.entity_set_id
                or np.setdiff1d(ids, np.asarray(boundary.entity_ids)).size
            ):
                issues.append(
                    "protected source faces bound to the exact original boundary entity set"
                )
            elif source.fidelity_source is None and (
                feature.feature_kind is FeatureKind.MATERIAL_INTERFACE
                or feature.maximum_deviation
                < min(
                    (
                        control.target_size
                        for control in specification.size_controls
                        if isinstance(control, sizing_owner.UniformSizeControl)
                    ),
                    default=np.inf,
                )
            ):
                issues.append("an independent source query for requested scoped fidelity")
        elif (
            dimension not in entity_ids
            or scope.source_id != specification.boundary_scope.source_id
            or scope.source_revision != specification.boundary_scope.source_revision
            or scope.entity_kind is not specification.boundary_scope.entity_kind
            or scope.entity_set_id != source.scope_entity_set_id(dimension)
            or np.setdiff1d(
                ids, entity_ids.get(dimension, np.empty((0,), dtype=np.int64))
            ).size
        ):
            issues.append("protected features outside the stable PLC entity banks")
    return issues


def _mapped_reference_support_issues(
    source: sources_owner.NativeLayerCoreSource, /
) -> list[str]:
    reference = source.reference_source
    domain = source.mapped_domain
    if reference is None:
        return (
            []
            if domain is None
            else ["the original reference source paired with its mapped domain"]
        )
    if not isinstance(domain, MappedReferenceDomain):
        return ["a typed MappedReferenceDomain for the mapped source layer/core"]
    issues: list[str] = []
    if (
        reference.reference_source is not None
        or reference.mapped_domain is not None
        or (reference.source_id, reference.source_revision)
        != (source.source_id, source.source_revision)
        or (domain.source_id, domain.source_revision)
        != (source.source_id, source.source_revision)
        or source.layers.control_id != reference.layers.control_id
        or domain.region_ids != reference.complex.region_ids
    ):
        issues.append("one nonrecursive source-bound reference PLC and mapped domain")

    def entity_bank(
        owner: sources_owner.NativeLayerCoreSource, dimension: int, /
    ) -> set[tuple[str, tuple[str, ...]]]:
        complex_ = owner.complex
        if dimension == 0:
            identifiers, paths = complex_.vertex_ids, complex_.vertex_occurrence_paths
        elif dimension == 1:
            identifiers, paths = complex_.segment_ids, complex_.segment_occurrence_paths
        elif dimension == 2:
            identifiers, paths = complex_.facet_ids, complex_.facet_occurrence_paths
        else:
            raise ValueError("PLC entity bank dimensions must be zero through two.")
        return set(zip(identifiers, paths, strict=True))

    if any(
        entity_bank(source, dimension) != entity_bank(reference, dimension)
        for dimension in (0, 1, 2)
    ):
        issues.append(
            "reference PLC vertices, segments and facets with original source identities and occurrences"
        )

    def facet_regions(
        owner: sources_owner.NativeLayerCoreSource, /
    ) -> dict[
        tuple[str, tuple[str, ...]],
        tuple[tuple[str, tuple[str, ...]] | None, ...],
    ]:
        complex_ = owner.complex
        region_keys = tuple(
            zip(complex_.region_ids, complex_.region_occurrence_paths, strict=True)
        )
        region_rows = np.asarray(complex_.facet_regions, dtype=np.int64)
        if region_rows.shape != (len(complex_.facet_ids), 2):
            raise ValueError(
                "Layer-core PLC facet regions must retain two oriented sides per facet."
            )
        return {
            (facet_id, path): tuple(
                None if int(region) < 0 else region_keys[int(region)] for region in row
            )
            for facet_id, path, row in zip(
                complex_.facet_ids,
                complex_.facet_occurrence_paths,
                region_rows.tolist(),
                strict=True,
            )
        }

    if facet_regions(source) != facet_regions(reference):
        issues.append(
            "reference PLC facet sides with identical authored region identities"
        )
    if (
        source.region_ids != reference.region_ids
        or source.region_occurrence_paths != reference.region_occurrence_paths
        or not np.array_equal(source.region_entity_ids, reference.region_entity_ids)
        or not np.array_equal(source.core_region_map, reference.core_region_map)
        or source.core_facet_source_ids != reference.core_facet_source_ids
        or source.core_facet_source_occurrence_paths
        != reference.core_facet_source_occurrence_paths
    ):
        issues.append("reference composite region and source-facet identity maps")
    return issues


@checked
def layer_core_support_issues(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:
    """Refuse only combinations whose exact fixed-layer/core semantics are unsupported."""
    issues: list[str] = []
    target = specification.target
    polyhedral = specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order not in (1, 2, 3)
    ):
        issues.append(
            "a three-dimensional volume map of supported degree one through three"
        )
    if polyhedral and target.geometry_order != 1:
        issues.append("first-order polyhedral core geometry")
    if polyhedral and (
        source.reference_source is not None or source.mapped_domain is not None
    ):
        issues.append(
            "an unmapped polyhedral core until mapped mixed-cell geometry is supported"
        )
    core_family = "polyhedron" if polyhedral else "tetrahedron"
    kinds = {
        "tetrahedron"
        if block.cell_kind in ("simplex:3", "tetrahedron")
        else block.cell_kind
        for block in source.layers.mesh.blocks
    } | {core_family}
    family = target.cell_families
    requested = set((*family.required, *family.preferred, *family.allowed_transitions))
    if (
        not set(family.required) <= kinds
        or not kinds <= requested
        or (len(kinds) > 1 and not family.allow_mixed)
        or (polyhedral and "polyhedron" not in family.required)
    ):
        issues.append("the requested exact layer/core cell-family policy")
    if specification.fill_strategy not in (
        VolumeFillStrategy.SIMPLEX,
        VolumeFillStrategy.POLYHEDRAL,
    ):
        issues.append("a native simplex or restricted-power core fill strategy")
    if polyhedral and source.core_power_source is None:
        issues.append("an authored core power source bound to the fixed PLC")
    if source.complex.boundary != "fixed":
        issues.append("an immutable fixed PLC core boundary")
    if source.complex.geometry is not None and source.reference_source is None:
        issues.append("a source-bound mapped reference domain for PLC coordinates")
    issues.extend(_boundary_control_issues(source, specification))
    issues.extend(controls_owner.layer_control_issues(source, specification))
    if (
        specification.periodic_constraints
        and source.layers.mesh.periodic_topology is None
    ):
        issues.append("source-authored periodic layers and core seam anatomy")
    try:
        realized = tuple(
            controls_owner.realized_layer_control(source, control).control_id
            for control in specification.layer_controls
        )
    except ValueError as error:
        issues.append(f"layer controls resolvable to the accepted layer wall: {error}")
    else:
        if any(identifier != source.layers.control_id for identifier in realized):
            issues.append(
                "layer controls different from the supplied accepted layer realization"
            )
    if any(
        _core_region_row(source, specification, control) is None
        for control in specification.region_controls
    ):
        issues.append("region controls bound to exact authored composite region IDs")
    issues.extend(_mapped_reference_support_issues(source))
    query = source.fidelity_source
    if isinstance(query, MappedReferenceBoundarySource) and (
        source.mapped_domain is None
        or query.domain.domain_id != source.mapped_domain.domain_id
        or query.authority.domain.domain_id != source.mapped_domain.domain_id
    ):
        issues.append(
            "mapped-root boundary query bound to the original layer source domain"
        )
    issues.extend(_feature_support_issues(source, specification))
    if source.fidelity_source is None:
        issues.append("a retained source-bound boundary query for fidelity")
    if source.source_surface_charts is None and specification.target.geometry_order > 1:
        query = source.fidelity_source
        if isinstance(query, MappedReferenceBoundarySource):
            if (
                source.mapped_domain is None
                or query.domain.domain_id != source.mapped_domain.domain_id
            ):
                issues.append("the original mapped-root chart and layer source identity")
        else:
            from ...geometry.implicit._adaptive_discovery import (
                AdaptiveImplicitBoundarySource,
            )

            adaptive_query = (
                query
                if isinstance(query, AdaptiveImplicitBoundarySource)
                else getattr(query, "adaptive_source", None)
            )
            if not isinstance(adaptive_query, AdaptiveImplicitBoundarySource):
                issues.append(
                    "the original adaptive implicit root-scaffold authority for high-order layers"
                )
    if source.maximum_deviation is None and specification.maximum_deviation is None:
        issues.append("an explicit physical source-fidelity tolerance")
    if any(
        seed.region_name not in source.complex.region_ids
        for seed in specification.region_seeds
    ):
        issues.append("region seeds naming an absent authoritative core region")
    return issues


def _edge_records(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact edge identities, endpoint rows and physical lengths."""
    identifiers = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    row_by_id = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    edges = np.asarray(
        [
            [row_by_id[first], row_by_id[second]]
            for first, second in canonical_owner._entity_vertex_keys(mesh, 1)
        ],
        dtype=np.int64,
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    return identifiers, edges, lengths


def _cell_edge_records(
    mesh: CellMesh, cell_ids: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Measure unique physical edges incident to the selected stable cell IDs."""
    selected = np.asarray(cell_ids, dtype=np.int64)
    if (
        selected.ndim != 1
        or selected.size == 0
        or np.unique(selected).size != selected.size
    ):
        raise ValueError("Core cell identity must be a nonempty unique integer bank.")
    selected_ids = {int(identifier) for identifier in selected.tolist()}
    found: set[int] = set()
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    row_by_id = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    edge_keys: set[tuple[int, int]] = set()
    if isinstance(mesh.connectivity, PolyhedralConnectivity):
        cell_global_ids = np.asarray(mesh.connectivity.cell_global_ids, dtype=np.int64)
        cell_faces = layer_core_owner.layer_topology.polyhedral_cell_faces(mesh)
        for cell_row, raw_identifier in enumerate(cell_global_ids.tolist()):
            identifier = int(raw_identifier)
            if identifier not in selected_ids:
                continue
            found.add(identifier)
            for loop in cell_faces[cell_row]:
                for local, first_row in enumerate(loop):
                    second_row = loop[(local + 1) % len(loop)]
                    first = int(vertex_ids[first_row])
                    second = int(vertex_ids[second_row])
                    edge_keys.add((min(first, second), max(first, second)))
    else:
        for block in mesh.blocks:
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            vertices = np.asarray(block.vertices, dtype=np.int64)
            topology = reference_cell_topology(block.cell_kind)
            for cell_id, row in zip(identifiers.tolist(), vertices, strict=True):
                identifier = int(cell_id)
                if identifier not in selected_ids:
                    continue
                found.add(identifier)
                global_vertices = vertex_ids[row]
                for local_edge in topology.entities[1]:
                    first, second = (
                        int(value)
                        for value in global_vertices[
                            np.asarray(local_edge, dtype=np.int64)
                        ]
                    )
                    edge_keys.add((min(first, second), max(first, second)))
    if found != selected_ids:
        raise ValueError("Core cell IDs are not an exact subset of the combined mesh.")
    edges = np.asarray(
        [(row_by_id[first], row_by_id[second]) for first, second in sorted(edge_keys)],
        dtype=np.int64,
    ).reshape((-1, 2))
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    return edges, lengths


def _maximum_adjacent_growth(edges: np.ndarray, lengths: np.ndarray, /) -> float:
    incident: dict[int, list[float]] = {}
    for edge, length in zip(edges, lengths, strict=True):
        incident.setdefault(int(edge[0]), []).append(float(length))
        incident.setdefault(int(edge[1]), []).append(float(length))
    growth = 1.0
    for values in incident.values():
        smallest = min(values)
        if smallest == 0.0:
            return float(np.inf)
        growth = max(growth, max(values) / smallest)
    return growth


def _compliance_report(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    /,
    soft_feature_outcomes: tuple[tuple[ProtectedFeature, float, bool], ...] = (),
) -> result_owner.MeshingComplianceReport:
    """Measure physical core size, fixed layer thickness, and native convergence evidence."""
    size = specification.size_controls[0]
    if not isinstance(size, sizing_owner.UniformSizeControl):
        raise TypeError("Layer/core compliance requires a uniform size control.")
    _, _, all_lengths = _edge_records(construction.mesh)
    core_edges, core_lengths = _cell_edge_records(
        construction.mesh, construction.core_cell_ids
    )
    core_maximum = float(np.max(core_lengths))
    core_minimum = float(np.min(core_lengths))
    requested: list[tuple[str, float]] = [("uniform_target_size", size.target_size)]
    achieved: list[tuple[str, float]] = [
        ("maximum_edge_length", float(np.max(all_lengths))),
        ("minimum_edge_length", float(np.min(all_lengths))),
        ("core_maximum_edge_length", core_maximum),
        ("core_minimum_edge_length", core_minimum),
        ("core_cell_count", float(construction.core_cell_ids.size)),
        ("fixed_boundary_face_count", float(construction.core.boundary_faces.shape[0])),
    ]
    for feature, deviation_upper, warning in soft_feature_outcomes:
        requested.append(
            (
                f"soft_feature:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        achieved.extend(
            (
                (f"soft_feature:{feature.feature_id}:deviation_upper", deviation_upper),
                (f"soft_feature:{feature.feature_id}:warning", float(warning)),
            )
        )
    issues: list[str] = []
    tolerance = (
        specification.size_compliance.absolute_tolerance
        + specification.size_compliance.relative_tolerance * size.target_size
    )
    if size.maximum_growth_rate is not None:
        growth = _maximum_adjacent_growth(core_edges, core_lengths)
        requested.append(("maximum_growth_rate", size.maximum_growth_rate))
        achieved.append(("core_maximum_adjacent_edge_growth", growth))
        growth_tolerance = (
            specification.size_compliance.absolute_tolerance
            + specification.size_compliance.relative_tolerance * size.maximum_growth_rate
        )
        if (
            size.strength is sizing_owner.SizeControlStrength.HARD
            and growth > size.maximum_growth_rate + growth_tolerance
        ):
            issues.append("maximum_growth_rate")
    if size.strength is sizing_owner.SizeControlStrength.HARD:
        if size.maximum_size is not None and core_maximum > size.maximum_size + tolerance:
            issues.append("maximum_size")
        if core_maximum > size.target_size + tolerance:
            issues.append("uniform_target_size")
        if size.minimum_size is not None and core_minimum < size.minimum_size - tolerance:
            issues.append("minimum_size")
    for control in specification.layer_controls:
        if control.core_maximum_size is None:
            continue
        cap = control.core_maximum_size
        requested.append((f"layer:{control.control_id}:core_maximum_size", cap))
        achieved.append((f"layer:{control.control_id}:core_maximum_edge", core_maximum))
        if (
            core_maximum
            > cap
            + specification.size_compliance.absolute_tolerance
            + specification.size_compliance.relative_tolerance * cap
        ):
            issues.append(f"layer:{control.control_id}:core_maximum_size")
    layer_evidence = source.layers.evidence
    active = np.asarray(layer_evidence.layer_active, dtype=np.bool_)
    means = np.asarray(layer_evidence.achieved_thicknesses, dtype=np.float64)
    minima = np.asarray(layer_evidence.minimum_thicknesses, dtype=np.float64)
    maxima = np.asarray(layer_evidence.maximum_thicknesses, dtype=np.float64)
    if not (
        active.shape
        == means.shape
        == minima.shape
        == maxima.shape
        == (len(layer_evidence.requested_thicknesses),)
    ):
        raise ValueError("Accepted layer thickness evidence has incompatible extents.")
    for index, requested_thickness in enumerate(layer_evidence.requested_thicknesses):
        requested_thickness = float(requested_thickness)
        requested.append((f"layer:{index}:thickness", requested_thickness))
        achieved.append((f"layer:{index}:active", float(active[index])))
        if not active[index]:
            issues.append(f"layer:{index}:inactive")
            continue
        for metric, value in (
            ("mean_thickness", means[index]),
            ("minimum_thickness", minima[index]),
            ("maximum_thickness", maxima[index]),
        ):
            if not np.isfinite(value):
                issues.append(f"layer:{index}:nonfinite_{metric}")
            else:
                achieved.append((f"layer:{index}:{metric}", float(value)))
    for control in specification.layer_controls:
        requested.append(
            (
                f"layer:{control.control_id}:minimum_thickness_fraction",
                control.minimum_thickness_fraction,
            )
        )
        requested.append(
            (
                f"layer:{control.control_id}:maximum_corner_stretch",
                control.maximum_corner_stretch,
            )
        )
        minimum_scale = float(layer_evidence.minimum_scale)
        maximum_stretch = float(layer_evidence.maximum_stretch)
        for metric, value in (
            ("minimum_scale", minimum_scale),
            ("maximum_stretch", maximum_stretch),
        ):
            if np.isfinite(value):
                achieved.append((f"layer:{control.control_id}:{metric}", value))
            else:
                issues.append(f"layer:{control.control_id}:nonfinite_{metric}")
        achieved.extend(
            (
                (
                    f"layer:{control.control_id}:detected_collision_count",
                    float(layer_evidence.detected_collision_count),
                ),
                (
                    f"layer:{control.control_id}:reduced_vertex_count",
                    float(layer_evidence.reduced_vertex_count),
                ),
                (
                    f"layer:{control.control_id}:terminated_vertex_count",
                    float(layer_evidence.terminated_vertex_count),
                ),
                (
                    f"layer:{control.control_id}:merged_vertex_count",
                    float(layer_evidence.merged_vertex_count),
                ),
            )
        )
        if not np.isfinite(minimum_scale) or minimum_scale < (
            control.minimum_thickness_fraction
            - specification.size_compliance.absolute_tolerance
            - specification.size_compliance.relative_tolerance
            * control.minimum_thickness_fraction
        ):
            issues.append(f"layer:{control.control_id}:minimum_thickness_fraction")
        if not np.isfinite(maximum_stretch) or maximum_stretch > (
            control.maximum_corner_stretch
            + specification.size_compliance.absolute_tolerance
            + specification.size_compliance.relative_tolerance
            * control.maximum_corner_stretch
        ):
            issues.append(f"layer:{control.control_id}:maximum_corner_stretch")
        if control.growth_rate_bounds is not None:
            lower, upper = control.growth_rate_bounds
            growth = np.asarray(layer_evidence.achieved_growth_rates, dtype=np.float64)
            requested.extend(
                (
                    (f"layer:{control.control_id}:minimum_growth_rate", lower),
                    (f"layer:{control.control_id}:maximum_growth_rate", upper),
                )
            )
            for index, value in enumerate(growth):
                if not np.isfinite(value):
                    if active[index] and active[index + 1]:
                        issues.append(f"layer:{control.control_id}:growth_rate:{index}")
                    continue
                achieved.append(
                    (f"layer:{control.control_id}:growth_rate:{index}", float(value))
                )
                growth_tolerance = (
                    specification.size_compliance.absolute_tolerance
                    + specification.size_compliance.relative_tolerance
                    * max(abs(lower), abs(upper))
                )
                if value < lower - growth_tolerance or value > upper + growth_tolerance:
                    issues.append(f"layer:{control.control_id}:growth_rate:{index}")
    if layer_evidence.certified_valid_count != sum(
        block.cell_count for block in source.layers.mesh.blocks
    ):
        issues.append("layer:certified_valid_cell_count")
    if not np.isfinite(layer_evidence.minimum_visibility):
        issues.append("layer:minimum_visibility")
    requested.extend(construction.periodic_requested)
    achieved.extend(construction.periodic_achieved)
    if construction.polyhedral_core is None:
        # A restricted-power core's tetrahedral carrier is an internal scaffold;
        # its explicit quality request is enforced by the power generator and
        # output cells are measured above, so carrier refinement is not output.
        # As in the fixed-boundary PLC route, refinement is construction evidence:
        # the fixed cap and source facets cannot be split, and the accepted cell
        # quality is the improvement outcome.
        achieved.extend(
            (
                (
                    "native_refinement_status",
                    float(int(construction.core.refinement_status)),
                ),
                (
                    "native_refinement_unmet_count",
                    float(len(construction.core.refinement_unmet)),
                ),
            )
        )
        if construction.core.improvement_status is not MeshcoreStatus.OK:
            issues.append("native_improvement_status")
        for name, _, _, _ in construction.core.improvement_unmet:
            issues.append(f"native_improvement:{name}")
    return result_owner.MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(requested),
        achieved=tuple(achieved),
        issues=tuple(issues),
    )


_ExactPoint2 = tuple[Fraction, Fraction]
# A moved or strip vertex resolved on (patch, output row, seed chart parameters).
_Request = tuple[int, int, _ExactPoint2]
# Relative chart distance within which a shared column vertex lies on a trim.
_TRIM_SNAP_TOLERANCE = 1.0e-9


def _layer_surface_chart_witness(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    query: MeshingDomainBoundarySource,
    /,
) -> SourceBoundaryChartWitness:
    """Bind unchanged input-wall chart pieces to their exact output boundary facets."""
    charts = source.source_surface_charts
    if charts is None or source.fidelity_source is None:
        raise ValueError(
            "Layer surface fidelity requires retained root charts and query."
        )
    charts.require_root(source.layers.source_wall, charts.geometry)
    charts.require_bound(
        source.layers.source_wall, charts.geometry, source.fidelity_source
    )
    if (
        query.source_id != charts.source_id
        or query.source_revision != charts.source_revision
        or query.domain.domain_id != charts.domain.domain_id
    ):
        raise ValueError("Layer scoped boundary query changes the retained chart source.")

    mesh = construction.mesh
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Layer surface chart witnesses require a 3D volume mesh.")
    wall_map = np.asarray(source.layers.wall_vertices, dtype=np.int64)
    source_cell_vertices: dict[int, np.ndarray] = {}
    for block in charts.mesh.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Layer source charts require triangular root cells.")
        for cell_id, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            source_cell_vertices[int(cell_id)] = np.asarray(vertices, dtype=np.int64)

    output_faces = _layer_output_boundary_faces(mesh)
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    # An open wall binds only its grown cells unchanged; the fixed remainder and
    # the layer side strips are rebound by ``_open_wall_bindings``.
    grown = (
        None
        if source.layers.closed_cap
        else set(brep_owner.grown_wall_cells(source.layers).tolist())
    )
    bindings: list[tuple[tuple[int, ...], dict[int, _ExactPoint2], int]] = []
    for chart_row, cell_id in enumerate(charts.cell_global_ids):
        patch = int(charts.root_patches[chart_row])
        if grown is not None and int(cell_id) not in grown:
            continue
        root_vertices = source_cell_vertices.get(int(cell_id))
        if root_vertices is None:
            raise ValueError("Layer root chart names an absent source-wall cell.")
        if np.any(root_vertices >= wall_map.size):
            raise ValueError(
                "Layer root chart vertex is outside the retained source wall."
            )
        mapped_vertices = wall_map[root_vertices]
        if np.any(mapped_vertices < 0):
            raise ValueError("Layer source-wall ancestry omits a root chart vertex.")
        mapped_ids = target_vertex_ids[mapped_vertices]
        exact_root_parameters = charts.root_parameter_chart(chart_row)
        if len(exact_root_parameters) != len(mapped_ids):
            raise ValueError(
                "Layer root chart parameters do not align with source vertices."
            )
        bindings.append(
            (
                tuple(sorted(int(value) for value in mapped_ids)),
                {
                    int(vertex_id): exact_root_parameters[corner]
                    for corner, vertex_id in enumerate(mapped_ids.tolist())
                },
                patch,
            )
        )
    if grown is not None:
        bindings.extend(
            _open_wall_bindings(
                source,
                construction,
                query,
                output_faces,
                source_cell_vertices,
                {key for key, _, _ in bindings},
            )
        )

    source_parameters: list[np.ndarray] = []
    source_parameter_rational_rows = []
    target_reference_rational_rows = []
    target_reference_vertices: list[np.ndarray] = []
    target_cell_ids: list[int] = []
    target_face_ids: list[int] = []
    target_local_faces: list[int] = []
    source_chart_ids: list[str] = []
    source_entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    orientations: list[int] = []
    for face_key, parameter_by_vertex, patch in bindings:
        if patch not in query.patches:
            continue
        candidates = output_faces.get(face_key, ())
        if len(candidates) != 1:
            raise ValueError(
                "A layer root chart does not bind one unique exterior target facet."
            )
        face_id, cell_global_id, local_face, _, face_vertex_ids, reference_by_vertex = (
            candidates[0]
        )
        if len(face_vertex_ids) == 3:
            triangles = (face_vertex_ids,)
        elif len(face_vertex_ids) == 4:
            # Quadrilateral layer side faces split in their local face order.
            first, second, third, fourth = face_vertex_ids
            triangles = ((first, second, third), (first, third, fourth))
        else:
            raise ValueError(
                "Layer boundary charts bind triangle and quadrilateral faces."
            )
        entity_id = query.domain.entity_id(2, patch)
        occurrence = tuple(query.domain.source_occurrences[2][patch])
        for triangle in triangles:
            target_uv_exact = tuple(parameter_by_vertex[vertex] for vertex in triangle)
            target_uv = np.asarray(
                [[float(value) for value in point] for point in target_uv_exact],
                dtype=np.float64,
            )
            signed_area = float(
                (target_uv[1, 0] - target_uv[0, 0]) * (target_uv[2, 1] - target_uv[0, 1])
                - (target_uv[1, 1] - target_uv[0, 1])
                * (target_uv[2, 0] - target_uv[0, 0])
            )
            if signed_area == 0.0:
                raise ValueError(
                    "Layer source chart collapses in its original parameter plane."
                )
            target_reference_exact = tuple(
                reference_by_vertex[vertex] for vertex in triangle
            )
            source_parameters.append(target_uv)
            source_parameter_rational_rows.append(target_uv_exact)
            target_reference_vertices.append(
                np.asarray(target_reference_exact, dtype=np.float64)
            )
            target_reference_rational_rows.append(target_reference_exact)
            target_cell_ids.append(cell_global_id)
            target_face_ids.append(face_id)
            target_local_faces.append(local_face)
            source_chart_ids.append(entity_id)
            source_entity_ids.append(entity_id)
            occurrence_paths.append(occurrence)
            orientations.append(1 if signed_area > 0.0 else -1)

    if not target_cell_ids:
        raise ValueError("Layer source chart query has no realized target facets.")
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(construction.geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_scope_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(target_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(target_face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(target_local_faces, dtype=np.int32),
        target_reference_vertices=np.asarray(target_reference_vertices, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            target_reference_rational_rows
        ),
        source_chart_ids=tuple(source_chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=np.asarray(source_parameters, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rational_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    source_fidelity_owner.validate_source_fidelity_witness(
        mesh, construction.geometry, query, witness
    )
    return witness


def _snap_to_trim(
    domain: MeshingDomain, patch: int, parameters: np.ndarray, /
) -> _ExactPoint2:
    """Place chart parameters exactly on the nearest straight trim segment."""
    polygon = _patch_chart_polygon(domain, patch)
    if polygon is None:
        raise ValueError(
            "A layer column shared by two source charts needs straight exact trims."
        )
    points, segments, _ = polygon
    corners = np.asarray(points, dtype=np.float64)
    extent = max(1.0, float(np.max(np.ptp(corners, axis=0))))
    start = corners[[first for first, _ in segments]]
    span = corners[[second for _, second in segments]] - start
    length = np.maximum(np.sum(span * span, axis=1), np.finfo(np.float64).tiny)
    step = np.clip(np.sum((parameters - start) * span, axis=1) / length, 0.0, 1.0)
    distance = np.linalg.norm(start + step[:, None] * span - parameters, axis=1)
    nearest = int(np.argmin(distance))
    if distance[nearest] > _TRIM_SNAP_TOLERANCE * extent:
        raise ValueError(
            "A layer column shared by two source charts leaves their common trim "
            f"(chart distance {float(distance[nearest]):.3g})."
        )
    first, second = (points[index] for index in segments[nearest])
    point = tuple(Fraction.from_float(float(value)) for value in parameters)
    direction = (second[0] - first[0], second[1] - first[1])
    fraction = (
        (point[0] - first[0]) * direction[0] + (point[1] - first[1]) * direction[1]
    ) / (direction[0] * direction[0] + direction[1] * direction[1])
    fraction = min(max(fraction, Fraction(0)), Fraction(1))
    return (
        first[0] + fraction * direction[0],
        first[1] + fraction * direction[1],
    )


def _open_wall_bindings(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    query: MeshingDomainBoundarySource,
    output_faces: Mapping[tuple[int, ...], list[_LayerOutputFace]],
    source_cell_vertices: Mapping[int, np.ndarray],
    grown_keys: set[tuple[int, ...]],
    /,
) -> list[tuple[tuple[int, ...], dict[int, _ExactPoint2], int]]:
    """Bind the fixed core surfaces and layer side strips of an open wall.

    Fixed surface cells keep their exact charts except at rim vertices, which
    moved to their column tops; side strips lie on the adjacent source faces.
    Moved and strip vertices take the source chart parameters of their closest
    point on that face, seeded at the rim vertex; a vertex shared by two charts
    lies exactly on their common straight trim.
    """
    charts = source.source_surface_charts
    domain = source.source_domain
    if charts is None or not isinstance(domain, MeshingDomain):
        raise ValueError("Open layer walls require their retained charted BRep source.")
    if domain.domain_id != query.domain.domain_id:
        raise ValueError("Open layer wall charts must bind the queried source domain.")
    layers = source.layers
    core = brep_owner.fixed_core(domain, layers, charts)
    if core.complex_.complex_id != source.complex.complex_id:
        raise ValueError(
            "An open layer wall must keep the fixed core derived from its charted wall."
        )
    combined_map = layer_core_owner.layer_topology.prepare_identity(
        layers, source.complex, source.vertex_layer_ids, source.cap_polygon_ids
    )[0]
    mesh = construction.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if not layer_core_owner.layer_topology.bitwise_equal_points(
        points[combined_map], np.asarray(source.complex.vertices, dtype=np.float64)
    ):
        raise ValueError("The open layer core output moved a fixed PLC vertex.")
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    row_of_target = {int(value): row for row, value in enumerate(target_ids.tolist())}
    wall_ids = np.asarray(charts.mesh.vertex_global_ids, dtype=np.int64)
    wall_row_of_id = {int(value): row for row, value in enumerate(wall_ids.tolist())}
    wall_map = np.asarray(layers.wall_vertices, dtype=np.int64)
    layer_count = int(np.asarray(layers.mesh.coordinates).shape[0])
    layer_sources = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    chart_row_of_cell = {
        int(cell): row for row, cell in enumerate(np.asarray(charts.cell_global_ids))
    }
    grown = set(brep_owner.grown_wall_cells(layers).tolist())

    def cell_chart(cell: int, /) -> tuple[int, dict[int, _ExactPoint2]]:
        row = chart_row_of_cell[cell]
        exact = charts.root_parameter_chart(row)
        return int(charts.root_patches[row]), {
            int(wall_row): exact[corner]
            for corner, wall_row in enumerate(source_cell_vertices[cell].tolist())
        }

    def source_wall_row(layer_row: int, /) -> int:
        identifier = int(layer_sources[layer_row]) if layer_row < layer_count else -1
        wall_row = wall_row_of_id.get(identifier)
        if wall_row is None:
            raise ValueError("An open layer wall vertex has no source wall vertex.")
        return wall_row

    # Exact parameters per face, plus moved/strip vertices resolved once per
    # (patch, output row, seed chart) by projection onto the source face.
    raw: list[
        tuple[tuple[int, ...], dict[int, _ExactPoint2], dict[int, _Request], int]
    ] = []
    requests: dict[_Request, None] = {}
    polygons = np.asarray(core.complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    for polygon_row in range(core.cap_count, polygons.shape[0]):
        cell = int(core.polygon_wall_cells[polygon_row])
        patch, corner_uv = cell_chart(cell)
        exact: dict[int, _ExactPoint2] = {}
        moved: dict[int, _Request] = {}
        for vertex in polygons[polygon_row].tolist():
            row = int(combined_map[vertex])
            wall_row = int(core.vertex_wall_rows[vertex])
            if wall_row >= 0:
                exact[int(target_ids[row])] = corner_uv[wall_row]
            else:
                key = (
                    patch,
                    row,
                    corner_uv[source_wall_row(int(core.vertex_layer_ids[vertex]))],
                )
                requests[key] = None
                moved[int(target_ids[row])] = key
        raw.append((tuple(sorted((*exact, *moved))), exact, moved, patch))
    fixed_keys = {key for key, _, _, _ in raw}

    rim_cells: dict[tuple[int, int], list[int]] = {}
    for cell, vertices in source_cell_vertices.items():
        if cell in grown:
            continue
        for corner in range(3):
            edge = sorted((int(vertices[corner]), int(vertices[(corner + 1) % 3])))
            rim_cells.setdefault((edge[0], edge[1]), []).append(cell)
    for face_key in output_faces:
        if face_key in grown_keys or face_key in fixed_keys:
            continue
        rows = [row_of_target[identifier] for identifier in face_key]
        if any(row >= layer_count for row in rows):
            raise ValueError(
                "An open layer boundary face is neither wall, core, nor strip."
            )
        sources = sorted({source_wall_row(row) for row in rows})
        cells = rim_cells.get((sources[0], sources[-1]), []) if len(sources) == 2 else []
        if len(cells) != 1:
            raise ValueError(
                "A layer side face must rise from one rim edge of one fixed surface cell."
            )
        patch, corner_uv = cell_chart(cells[0])
        exact = {}
        moved = {}
        for row in rows:
            wall_row = source_wall_row(row)
            if int(wall_map[wall_row]) == row:
                exact[int(target_ids[row])] = corner_uv[wall_row]
            else:
                key = (patch, row, corner_uv[wall_row])
                requests[key] = None
                moved[int(target_ids[row])] = key
        raw.append((face_key, exact, moved, patch))

    keys = list(requests)
    resolved: dict[_Request, _ExactPoint2] = {}
    if keys:
        projection = domain.project(
            points[[row for _, row, _ in keys]],
            np.asarray([patch for patch, _, _ in keys], dtype=np.int64),
            np.asarray(
                [[float(seed[0]), float(seed[1])] for _, _, seed in keys],
                dtype=np.float64,
            ),
        )
        if not np.all(np.asarray(projection.converged, dtype=np.bool_)):
            raise ValueError(
                "A layer side vertex has no closest point on its source face."
            )
        projected = np.asarray(projection.parameters, dtype=np.float64)
        shared = Counter(row for _, row, _ in keys)
        for index, key in enumerate(keys):
            resolved[key] = (
                _snap_to_trim(domain, key[0], projected[index])
                if shared[key[1]] > 1
                else (
                    Fraction.from_float(float(projected[index, 0])),
                    Fraction.from_float(float(projected[index, 1])),
                )
            )
    return [
        (
            face_key,
            {
                **exact,
                **{identifier: resolved[key] for identifier, key in moved.items()},
            },
            patch,
        )
        for face_key, exact, moved, patch in raw
    ]


_LayerOutputFace = tuple[
    int,
    int,
    int,
    str,
    tuple[int, ...],
    dict[int, tuple[Fraction, ...]],
]


def _layer_output_boundary_faces(
    mesh: CellMesh, /
) -> dict[tuple[int, ...], list[_LayerOutputFace]]:
    """Index true output boundary facets by their stable vertex identities."""
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Layer boundary charts require a 3D volume mesh.")
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    boundary = np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        face_vertices = tuple(
            np.asarray(row, dtype=np.int64) for row in connectivity.entities[2]
        )
    elif isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)):
        face_vertices = tuple(
            np.asarray(row, dtype=np.int64) for row in connectivity.faces
        )
    elif isinstance(connectivity, PolyhedralConnectivity):
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        face_vertices = tuple(
            values[int(offsets[row]) : int(offsets[row + 1])]
            for row in range(offsets.size - 1)
        )
    else:
        raise ValueError("Layer output has no canonical volume-face incidence.")
    if len(face_vertices) != face_ids.size or boundary.shape != face_ids.shape:
        raise ValueError("Layer target face incidence is incomplete.")
    face_row_by_vertices: dict[tuple[int, ...], int] = {}
    for row, face in enumerate(face_vertices):
        if (
            face.ndim != 1
            or face.size < 3
            or np.any(face < 0)
            or np.any(face >= target_vertex_ids.size)
        ):
            raise ValueError("Layer face incidence contains malformed vertex rows.")
        key = tuple(
            sorted(int(target_vertex_ids[int(vertex)]) for vertex in face.tolist())
        )
        if len(set(key)) != len(key) or key in face_row_by_vertices:
            raise ValueError(
                "Layer face incidence does not uniquely bind stable vertex identities."
            )
        face_row_by_vertices[key] = row
    result: dict[tuple[int, ...], list[_LayerOutputFace]] = {}
    for block in mesh.blocks:
        if block.cell_kind == "polyhedron":
            continue
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)
        block_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        topology = reference_cell_topology(block.cell_kind)
        if topology.dimension != 3:
            raise ValueError("Layer boundary charts require explicit volume-cell faces.")
        local_faces = topology.entities[2]
        reference_vertices = tuple(
            tuple(Fraction.from_float(float(value)) for value in point)
            for point in np.asarray(topology.vertices, dtype=np.float64)
        )
        for cell_row, cell_id in enumerate(block_cell_ids.tolist()):
            for local_face, local_vertices in enumerate(local_faces):
                vertex_rows = cell_vertices[
                    cell_row, np.asarray(local_vertices, dtype=np.int64)
                ]
                vertex_keys = tuple(
                    int(value) for value in target_vertex_ids[vertex_rows]
                )
                face_row = face_row_by_vertices.get(tuple(sorted(vertex_keys)))
                if face_row is None:
                    raise ValueError(
                        "Layer cell face is absent from canonical face incidence."
                    )
                if not boundary[face_row]:
                    continue
                parameter_by_vertex = {
                    identifier: reference_vertices[int(local_vertex)]
                    for identifier, local_vertex in zip(
                        vertex_keys, local_vertices, strict=True
                    )
                }
                result.setdefault(tuple(sorted(vertex_keys)), []).append(
                    (
                        int(face_ids[face_row]),
                        int(cell_id),
                        local_face,
                        block.cell_kind,
                        vertex_keys,
                        parameter_by_vertex,
                    )
                )
    return result


def _layer_cell_reference_identity(
    mesh: CellMesh, /
) -> dict[int, tuple[str, tuple[int, ...]]]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result = {}
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        for cell_id, cell in zip(
            np.asarray(block.global_ids, dtype=np.int64), vertices, strict=True
        ):
            identifier = int(cell_id)
            if identifier in result:
                raise ValueError("Layer reference cell IDs must be unique across blocks.")
            result[identifier] = (
                block.cell_kind,
                tuple(int(value) for value in vertex_ids[cell]),
            )
    return result


def reversed_triangle_wall(wall: CellMesh, /) -> CellMesh:
    """The same triangle wall with every cell's orientation reversed.

    Layers grow along the wall orientation. An adaptive implicit zero set is
    oriented out of its solid, so layers that grow into the solid use this
    reversal; cell, vertex and numeric identities are kept.
    """
    if len(wall.blocks) != 1 or wall.blocks[0].cell_kind != "triangle":
        raise ValueError("A reversed layer wall must be one triangle block.")
    block = wall.blocks[0]
    return CellMesh(
        wall.coordinates,
        (
            CellBlock(
                block.name,
                "triangle",
                np.asarray(block.vertices, dtype=np.int64)[:, (0, 2, 1)],
                global_ids=np.asarray(block.global_ids, dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(wall.vertex_global_ids, dtype=np.int64),
        numeric_version=wall.numeric_version,
    )


class ImplicitLayerWall(NamedTuple):
    """An implicit layer wall taken from the certified implicit carrier path."""

    wall: CellMesh
    """Closed triangle wall, oriented along the requested growth direction."""
    distance: float
    """Certified wall-to-zero-set distance bound."""
    evidence: tuple[tuple[str, float], ...]
    """Carrier quality before/after the zero-set remesh, and whether it ran."""


def _adaptive_layer_query(
    query: SourceBoundaryQuery, /
) -> AdaptiveImplicitBoundarySource:
    from ...geometry.implicit._adaptive_discovery import (
        AdaptiveImplicitBoundarySource,
    )

    adaptive_query = (
        query
        if isinstance(query, AdaptiveImplicitBoundarySource)
        else getattr(query, "adaptive_source", None)
    )
    if not isinstance(adaptive_query, AdaptiveImplicitBoundarySource):
        raise ValueError(
            "Implicit layer walls require the original adaptive boundary query."
        )
    adaptive_query.require_current()
    return adaptive_query


def implicit_layer_wall(
    query: SourceBoundaryQuery,
    /,
    *,
    radius_edge_bound: float = 2.0,
    edge_length: float | None = None,
    inward: bool = True,
) -> ImplicitLayerWall:
    """The implicit zero-set wall a layer route grows from.

    This is the implicit volume provider's carrier: the discovery scaffold
    triangles when every one is within ``radius_edge_bound`` (the core fill
    schedule bound; a layer column inherits its wall triangle's shape), else
    the native zero-set band remesh toward ``edge_length`` (default: the
    scaffold's mean edge) with every created triangle certified within the
    scaffold's distance bound and transversal to the field gradient. Its root
    charts and boundary degree are proved downstream like the volume
    carrier's. Layers grow along the wall orientation; ``inward`` reverses the
    outward zero-set orientation so they grow into the solid.
    """
    from ...geometry._interval_enclosure import (
        IntervalExtensionError,
        prepare_interval_function,
    )
    from ._native_implicit import (
        _CARRIER_MINIMUM_ANGLE_DEGREES,
        _carrier_quality,
        _face_radius_edge_ratios,
    )

    if not (math.isfinite(radius_edge_bound) and radius_edge_bound >= 1.0):
        raise ValueError("radius_edge_bound must be a finite ratio of at least one.")
    if edge_length is not None and not (math.isfinite(edge_length) and edge_length > 0.0):
        raise ValueError("edge_length must be positive and finite when supplied.")
    adaptive_query = _adaptive_layer_query(query)
    discovery = adaptive_query.surface
    if discovery.mesh is None:
        raise ValueError("Implicit layer walls require a complete discovery mesh.")
    vertices = np.asarray(discovery.mesh.vertices, dtype=np.float64)
    faces = np.asarray(discovery.mesh.faces, dtype=np.int64)
    scaffold_distance = float(
        discovery.evidence.maximum_surface_box_diagonal + discovery.policy.root_tolerance
    )
    before = _carrier_quality(vertices, faces, "before")
    if float(np.max(_face_radius_edge_ratios(vertices, faces))) <= radius_edge_bound:
        distance = scaffold_distance
        evidence = (("carrier_remeshed", 0.0), *before)
    else:
        kernel, state = discovery.geometry.kernel, discovery.geometry.state

        def value(point: jax.Array) -> jax.Array:
            return kernel.boundary_field(state, point[None, :])[0]

        try:
            value_program, gradient_program = (
                prepare_interval_function(
                    function, 3, batch_capacity=1, execution="native"
                ).native_program
                for function in (value, jax.grad(value))
            )
        except IntervalExtensionError as error:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "The implicit layer wall remesh requires a native interval lowering "
                f"of the field and its gradient: {error}",
                provider_code="native_layer_implicit_wall",
                stage="source_inspection",
            ) from error
        if value_program is None or gradient_program is None:
            raise RuntimeError("Native interval preparation returned no native program.")
        if edge_length is None:
            edges = np.unique(
                np.sort(
                    np.concatenate(
                        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])
                    ),
                    axis=1,
                ),
                axis=0,
            )
            edge_length = float(
                np.mean(
                    np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
                )
            )
        try:
            remeshed = zero_set_remesh(
                vertices,
                faces,
                value_program.box_program(),
                gradient_program.box_program(),
                scaffold_distance,
                scaffold_distance,
                edge_length,
                _CARRIER_MINIMUM_ANGLE_DEGREES,
            )
        except (MeshcoreError, ValueError, OverflowError) as error:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"The implicit layer wall zero-set remesh could not be certified: {error}",
                provider_code="native_layer_implicit_wall",
                stage="surface_meshing",
            ) from error
        vertices = np.array(remeshed.vertices, dtype=np.float64, copy=True)
        faces = np.array(remeshed.triangles, dtype=np.int64, copy=True)
        worst = float(np.max(_face_radius_edge_ratios(vertices, faces)))
        if not worst <= radius_edge_bound:
            raise MeshingFailure(
                MeshingFailureCategory.QUALITY_REJECTED,
                "The certified zero-set remesh of the implicit layer wall leaves a "
                f"triangle with radius-edge ratio {worst:.6g} above "
                f"{radius_edge_bound:.6g}.",
                provider_code="native_layer_implicit_wall",
                stage="surface_meshing",
            )
        distance = float(remeshed.distance)
        evidence = (
            ("carrier_remeshed", 1.0),
            *before,
            *_carrier_quality(vertices, faces, "after"),
            ("carrier_remesh_edge_length", edge_length),
            ("carrier_remesh_unmet_count", float(remeshed.counters[4])),
        )
    wall = CellMesh.from_triangles(vertices, faces, numeric_version=query.source_revision)
    return ImplicitLayerWall(
        reversed_triangle_wall(wall) if inward else wall, distance, evidence
    )


def _implicit_carrier_root_witness(
    source_id: str,
    source_revision: str,
    query: SourceBoundaryQuery,
    adaptive_query: AdaptiveImplicitBoundarySource,
    wall: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> SourceBoundaryChartWitness:
    """Root charts of a remeshed implicit wall, seeded like the volume carrier's.

    Each cell names the scaffold triangle with the nearest centroid. Corners
    that are not scaffold roots lift from their points, and repeated or omitted
    scaffold triangles leave the boundary degree to the covering proof. Cells
    oriented against the field gradient (an inward wall) list their corners
    reversed, so every chart keeps the zero-set orientation; the chart proofs
    certify that orientation.
    """
    from ._native_implicit import _nearest_scaffold_rows

    discovery = adaptive_query.surface
    scaffold_mesh = discovery.mesh
    if (
        scaffold_mesh is None
        or (query.source_id, query.source_revision) != (source_id, source_revision)
        or wall.numeric_version != source_revision
        or wall.topological_dimension != 2
        or wall.ambient_dimension != 3
        or len(wall.blocks) != 1
        or wall.blocks[0].cell_kind != "triangle"
    ):
        raise ValueError(
            "An implicit carrier wall must be one triangle block of the current "
            "source revision."
        )
    block = wall.blocks[0]
    points = np.asarray(wall.coordinates, dtype=np.float64)
    faces = np.asarray(block.vertices, dtype=np.int64)
    triangles = points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    kernel, state = discovery.geometry.kernel, discovery.geometry.state

    def field(point: jax.Array) -> jax.Array:
        return kernel.boundary_field(state, point[None, :])[0]

    gradients = np.asarray(
        jax.vmap(jax.grad(field))(np.mean(triangles, axis=1)), dtype=np.float64
    )
    signs = np.einsum("ij,ij->i", normals, gradients)
    if not (np.all(signs > 0.0) or np.all(signs < 0.0)):
        raise ValueError(
            "An implicit carrier wall must be oriented consistently along or "
            "against the field gradient."
        )
    outward = bool(signs[0] > 0.0)
    rows = _nearest_scaffold_rows(
        np.asarray(scaffold_mesh.vertices, dtype=np.float64),
        np.asarray(scaffold_mesh.faces, dtype=np.int64),
        points,
        faces,
    )
    scaffold = discovery.scaffold
    cell_keys = np.asarray(scaffold.cell_indices, dtype=np.int64)[
        np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)[rows]
    ]
    root_edge_keys = np.asarray(scaffold.root_edge_ids, dtype=np.int64)[
        np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)[rows]
    ]
    chart_ids = tuple(
        source_fidelity_owner._implicit_chart_id(
            adaptive_query.source_id,
            adaptive_query.source_revision,
            (int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])),
            tuple(
                (
                    (int(edge[0, 0]), int(edge[0, 1]), int(edge[0, 2])),
                    (int(edge[1, 0]), int(edge[1, 1]), int(edge[1, 2])),
                )
                for edge in edges
            ),
        )
        for cell, edges in zip(cell_keys, root_edge_keys, strict=True)
    )
    count = faces.shape[0]
    standard = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    surface_entity_id = source_entity_id(
        source_id, source_revision, "implicit-surface", "zero-set"
    )
    return SourceBoundaryChartWitness(
        mesh_id=wall.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=wall.topology_id,
        source_id=source_id,
        source_revision=source_revision,
        source_query_id=adaptive_query.source_revision,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=cell_ids,
        entity_global_ids=cell_ids,
        local_entity_indices=np.full(count, -1, dtype=np.int32),
        target_reference_vertices=np.broadcast_to(
            standard if outward else standard[(0, 2, 1), :], (count, 3, 2)
        ),
        source_chart_ids=chart_ids,
        source_entity_ids=(surface_entity_id,) * count,
        occurrence_paths=((),) * count,
        source_parameters=np.broadcast_to(standard, (count, 3, 2)),
        orientations=np.full(count, 1 if outward else -1, dtype=np.int8),
        implicit_cell_keys=cell_keys,
        implicit_root_edge_keys=root_edge_keys,
    )


def _implicit_layer_root_witness(
    source_id: str,
    source_revision: str,
    query: SourceBoundaryQuery,
    wall: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> SourceBoundaryChartWitness:
    """Bind the adaptive root charts to an implicit layer wall.

    The wall is the exact discovery mesh (layers grow out of the solid), its
    reversal (layers grow into it), or a zero-set remeshed carrier from
    `implicit_layer_wall`. A reversed scaffold cell names the same scaffold
    triangle; its chart lists the corners in discovery order, so the target
    reference corners are the cell's own reference vertices permuted and the
    standard root parameters, cell keys and root edges are unchanged.
    """
    from ._native_implicit import _implicit_layer_wall_chart_witness

    adaptive_query = _adaptive_layer_query(query)
    discovered = adaptive_query.surface.mesh
    if (
        discovered is None
        or len(wall.blocks) != 1
        or wall.blocks[0].cell_kind != "triangle"
    ):
        return _implicit_layer_wall_chart_witness(
            source_id, source_revision, query, wall, geometry
        )
    if not np.array_equal(
        np.asarray(wall.coordinates, dtype=np.float64),
        np.asarray(discovered.vertices, dtype=np.float64),
    ):
        return _implicit_carrier_root_witness(
            source_id, source_revision, query, adaptive_query, wall, geometry
        )
    if not np.array_equal(
        np.asarray(wall.blocks[0].vertices, dtype=np.int64)[:, (0, 2, 1)],
        np.asarray(discovered.faces, dtype=np.int64),
    ):
        return _implicit_layer_wall_chart_witness(
            source_id, source_revision, query, wall, geometry
        )
    outward_wall = reversed_triangle_wall(wall)
    outward = _implicit_layer_wall_chart_witness(
        source_id,
        source_revision,
        query,
        outward_wall,
        CellGeometrySpec.affine(outward_wall),
    )
    corners = np.asarray(outward.target_reference_vertices, dtype=np.float64)[
        :, (0, 2, 1)
    ]
    return SourceBoundaryChartWitness(
        mesh_id=wall.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=wall.topology_id,
        source_id=outward.source_id,
        source_revision=outward.source_revision,
        source_query_id=outward.source_query_id,
        chart_dimension=outward.chart_dimension,
        reference_dimension=outward.reference_dimension,
        cell_global_ids=outward.cell_global_ids,
        entity_global_ids=outward.entity_global_ids,
        local_entity_indices=outward.local_entity_indices,
        target_reference_vertices=corners,
        source_chart_ids=outward.source_chart_ids,
        source_entity_ids=outward.source_entity_ids,
        occurrence_paths=outward.occurrence_paths,
        source_parameters=outward.source_parameters,
        source_parameter_rational_tokens=outward.source_parameter_rational_tokens,
        orientations=-np.asarray(outward.orientations, dtype=np.int8),
        implicit_cell_keys=outward.implicit_cell_keys,
        implicit_root_edge_keys=outward.implicit_root_edge_keys,
    )


def _implicit_layer_surface_chart_witness(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    /,
) -> SourceBoundaryChartWitness:
    """Project the original implicit wall charts onto their exact output facets."""
    query = source.fidelity_source
    if query is None:
        raise ValueError("Implicit layer fidelity requires its original boundary query.")
    wall = source.layers.source_wall
    root_witness = _implicit_layer_root_witness(
        source.source_id,
        source.source_revision,
        query,
        wall,
        CellGeometrySpec.affine(wall),
    )
    mesh = construction.mesh
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("Implicit layer boundary charts require a 3D volume mesh.")
    wall_map = np.asarray(source.layers.wall_vertices, dtype=np.int64)
    wall_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if (
        wall_map.shape != wall_ids.shape
        or np.any(wall_map < 0)
        or np.any(wall_map >= target_ids.size)
    ):
        raise ValueError("Implicit layer wall-to-output vertex incidence is incomplete.")
    reference_faces = _layer_output_boundary_faces(mesh)

    source_cells: dict[int, np.ndarray] = {}
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Implicit source wall must retain its triangle cells.")
        for cell_id, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            source_cells[int(cell_id)] = np.asarray(vertices, dtype=np.int64)

    root_cell_ids = np.asarray(root_witness.cell_global_ids, dtype=np.int64)
    if (
        len(source_cells) != root_cell_ids.size
        or len(set(root_cell_ids.tolist())) != root_cell_ids.size
    ):
        raise ValueError(
            "Implicit root witness must cover each original wall triangle once."
        )
    root_row_by_cell = {
        int(identifier): row for row, identifier in enumerate(root_cell_ids)
    }
    source_parameters = []
    source_parameter_tokens = []
    target_reference = []
    target_reference_tokens = []
    output_cell_ids = []
    output_face_ids = []
    output_local_faces = []
    orientations = []
    mapped_face_ids: set[int] = set()
    for source_cell_id, source_vertices in source_cells.items():
        root_row = root_row_by_cell.get(source_cell_id)
        if root_row is None:
            raise ValueError("Implicit wall cell is absent from its source root witness.")
        mapped_rows = wall_map[
            np.asarray(
                layer_curving_owner._implicit_root_corner_rows(
                    root_witness, root_row, source_vertices
                ),
                dtype=np.int64,
            )
        ]
        mapped_vertex_ids = tuple(int(value) for value in target_ids[mapped_rows])
        candidates = reference_faces.get(tuple(sorted(mapped_vertex_ids)), ())
        if len(candidates) != 1:
            raise ValueError(
                "Each implicit wall triangle must bind one unique output boundary facet."
            )
        face_id, cell_id, local_face, _, face_order, parameter_by_vertex = candidates[0]
        if face_id in mapped_face_ids:
            raise ValueError(
                "Two implicit source triangles bind one output boundary facet."
            )
        mapped_face_ids.add(face_id)
        target_chart = tuple(
            parameter_by_vertex[identifier] for identifier in mapped_vertex_ids
        )
        permutation = tuple(
            face_order.index(identifier) for identifier in mapped_vertex_ids
        )
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        target_reference.append(target_chart)
        target_reference_tokens.append(target_chart)
        source_parameters.append(root_witness.source_parameter_chart(root_row))
        source_parameter_tokens.append(root_witness.source_parameter_chart(root_row))
        output_cell_ids.append(cell_id)
        output_face_ids.append(face_id)
        output_local_faces.append(local_face)
        orientations.append(-1 if inversions % 2 else 1)

    if len(mapped_face_ids) != len(source_cells):
        raise ValueError("Implicit source wall ancestry omits an output boundary facet.")
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(construction.geometry),
        topology_id=mesh.topology_id,
        source_id=root_witness.source_id,
        source_revision=root_witness.source_revision,
        source_query_id=root_witness.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(output_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(output_face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(output_local_faces, dtype=np.int32),
        target_reference_vertices=np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in target_reference
            ],
            dtype=np.float64,
        ),
        target_reference_rational_tokens=rational_chart_tokens(target_reference_tokens),
        source_chart_ids=root_witness.source_chart_ids,
        source_entity_ids=root_witness.source_entity_ids,
        occurrence_paths=root_witness.occurrence_paths,
        source_parameters=np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in source_parameters
            ],
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=rational_chart_tokens(source_parameter_tokens),
        orientations=np.asarray(orientations, dtype=np.int8),
        implicit_cell_keys=np.asarray(root_witness.implicit_cell_keys),
        implicit_root_edge_keys=np.asarray(root_witness.implicit_root_edge_keys),
    )
    source_fidelity_owner.validate_source_fidelity_witness(
        mesh, construction.geometry, query, witness
    )
    return witness


def _mapped_layer_reference_controls(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    query: MappedReferenceBoundarySource,
    /,
) -> tuple[
    Mapping[int, nested_reference_owner.ExactReferenceChartControl],
    dict[int, nested_reference_owner.NestedCellReferenceMap],
]:
    domain = source.mapped_domain
    if not isinstance(domain, MappedReferenceDomain):
        raise TypeError("Mapped layer charts require their original mapped-root domain.")

    authority = query.authority
    query.require_current()
    authority.validate_restored()
    reference_mesh = construction.reference_mesh
    reference_geometry = construction.reference_geometry
    if (
        (source.source_id, source.source_revision)
        != (domain.source_id, domain.source_revision)
        or (query.source_id, query.source_revision)
        != (domain.source_id, domain.source_revision)
        or authority.domain.domain_id != domain.domain_id
        or query.domain.domain_id != domain.domain_id
        or construction.domain.domain_id != domain.domain_id
        or construction.source_mesh is None
        or construction.source_mesh.mesh_id != domain.reference_mesh.mesh_id
        or construction.source_geometry is None
        or cell_geometry_id(construction.source_geometry) != domain.reference_geometry_id
        or cell_geometry_id(domain.source_geometry) != domain.reference_geometry_id
        or reference_mesh is None
        or reference_geometry is None
        or _layer_cell_reference_identity(reference_mesh)
        != _layer_cell_reference_identity(construction.mesh)
        or not np.array_equal(
            np.asarray(reference_mesh.vertex_global_ids),
            np.asarray(construction.mesh.vertex_global_ids),
        )
        or source.complex.region_ids != domain.region_ids
    ):
        raise ValueError(
            "Mapped layer reference charts change their original root authority."
        )

    controls = nested_reference_owner.exact_reference_chart_controls(
        domain,
        reference_mesh,
        reference_geometry,
        np.asarray(construction.cell_regions, dtype=np.int64),
    )
    controls_by_id = nested_reference_owner.reference_chart_controls_by_target_id(
        controls
    )
    target_ids = {
        int(value)
        for block in reference_mesh.blocks
        for value in np.asarray(block.global_ids, dtype=np.int64)
    }
    physical_target_ids = {
        int(value)
        for block in construction.mesh.blocks
        for value in np.asarray(block.global_ids, dtype=np.int64)
    }
    if set(controls_by_id) != target_ids or target_ids != physical_target_ids:
        raise ValueError(
            "Exact mapped layer controls do not cover every physical target cell."
        )

    retained: dict[int, tuple[object, ...]] = {}
    for witness in construction.reference_maps:
        exact_vertices = witness.source_reference_exact_vertices
        arguments = witness.source_reference_chart_arguments
        target_cell_ids = np.asarray(witness.target_cell_ids, dtype=np.int64)
        source_cell_ids = np.asarray(witness.source_cell_ids, dtype=np.int64)
        if (
            exact_vertices is None
            or arguments is None
            or target_cell_ids.ndim != 1
            or source_cell_ids.shape != target_cell_ids.shape
            or len(exact_vertices) != target_cell_ids.size
            or len(arguments) != target_cell_ids.size
        ):
            raise ValueError(
                "Mapped layer construction lost exact source reference expressions."
            )
        for row, target_id_raw in enumerate(target_cell_ids.tolist()):
            target_id = int(target_id_raw)
            if target_id in retained:
                raise ValueError(
                    "Mapped layer construction repeats a target reference map."
                )
            retained[target_id] = (
                int(source_cell_ids[row]),
                witness.target_cell_kind,
                witness.source_cell_kind,
                exact_vertices[row],
                tuple(arguments[row]),
            )
    if set(retained) != target_ids:
        raise ValueError("Mapped layer reference maps omit or add a target cell.")

    nested_maps: dict[int, nested_reference_owner.NestedCellReferenceMap] = {}
    for target_id, control in controls_by_id.items():
        if (
            control.source_id != domain.source_id
            or control.source_revision != domain.source_revision
            or control.mapped_domain_id != domain.domain_id
            or control.source_geometry_id != domain.reference_geometry_id
            or control.source_topology_id != domain.reference_mesh.topology_id
            or control.source_region < 0
            or control.source_region >= len(domain.region_ids)
            or control.source_region_id != domain.region_ids[control.source_region]
        ):
            raise ValueError(
                "Mapped layer control changes source, region, or root topology identity."
            )
        expected = (
            int(control.source_cell_id),
            control.target_cell_kind,
            control.source_cell_kind,
            control.source_reference_vertices,
            tuple(control.arguments),
        )
        if retained[target_id] != expected:
            raise ValueError(
                "Mapped layer exact controls differ from retained reference-map ancestry."
            )
        nested_maps[target_id] = nested_reference_owner.NestedCellReferenceMap(
            target_id,
            int(control.source_cell_id),
            control.target_cell_kind,
            control.source_cell_kind,
            control.source_reference_vertices,
            tuple(control.arguments),
        )
    return controls_by_id, nested_maps


def _mapped_layer_root_face_index(
    query: MappedReferenceBoundarySource, /
) -> dict[tuple[int, int], int]:
    relation = query.authority.root_relation
    root_faces = np.asarray(relation.polygon_root_face_global_ids, dtype=np.int64)
    root_cells = np.asarray(relation.polygon_root_cell_global_ids, dtype=np.int64)
    root_local_faces = np.asarray(
        relation.polygon_root_local_face_indices, dtype=np.int64
    )
    if (
        root_faces.ndim != 1
        or root_cells.shape != (root_faces.size, 2)
        or root_local_faces.shape != root_cells.shape
    ):
        raise ValueError("Mapped PLC root relation has malformed face/cell incidence.")
    result: dict[tuple[int, int], int] = {}
    for row, root_face in enumerate(root_faces.tolist()):
        for cell_raw, local_raw in zip(
            root_cells[row], root_local_faces[row], strict=True
        ):
            cell_id, local_face = int(cell_raw), int(local_raw)
            if cell_id < 0:
                continue
            if local_face < 0:
                raise ValueError(
                    "Mapped root face incidence has a missing local-face identity."
                )
            key = (cell_id, local_face)
            previous = result.setdefault(key, int(root_face))
            if previous != int(root_face):
                raise ValueError(
                    "Mapped root cell/local-face identity names multiple global faces."
                )
    return result


def _exact_fraction_point(values: tuple[object, ...], /) -> tuple[Fraction, ...]:
    result = []
    for value in values:
        if isinstance(value, Fraction):
            result.append(value)
        elif isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            result.append(Fraction(int(value)))
        else:
            raise TypeError(
                "Mapped reference chart coordinates must remain exact rationals."
            )
    return tuple(result)


def _exact_fraction_pair(values: tuple[object, ...], /) -> tuple[Fraction, Fraction]:
    point = _exact_fraction_point(values)
    if len(point) != 2:
        raise ValueError(
            "Mapped face chart points require exactly two rational coordinates."
        )
    return point[0], point[1]


def _cross2(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], /
) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


def _signed_chart_area(points: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
    return sum(
        (
            _cross2(points[row], points[(row + 1) % len(points)])
            for row in range(len(points))
        ),
        Fraction(0),
    )


def _triangle_chart_coordinates(
    point: tuple[Fraction, Fraction],
    triangle: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> tuple[Fraction, Fraction]:
    first = (triangle[1][0] - triangle[0][0], triangle[1][1] - triangle[0][1])
    second = (triangle[2][0] - triangle[0][0], triangle[2][1] - triangle[0][1])
    delta = (point[0] - triangle[0][0], point[1] - triangle[0][1])
    determinant = _cross2(first, second)
    if determinant == 0:
        raise ValueError("Mapped source or target triangle chart is singular.")
    return _cross2(delta, second) / determinant, _cross2(first, delta) / determinant


def _triangle_chart_map(
    points: tuple[tuple[Fraction, Fraction], ...],
    domain: tuple[tuple[Fraction, Fraction], ...],
    image: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    output = []
    for point in points:
        u, v = _triangle_chart_coordinates(point, domain)
        output.append(
            tuple(
                image[0][axis]
                + u * (image[1][axis] - image[0][axis])
                + v * (image[2][axis] - image[0][axis])
                for axis in range(len(image[0]))
            )
        )
    return tuple(output)


def _clip_exact_triangle(
    subject: tuple[tuple[Fraction, Fraction], ...],
    clip_triangle: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> tuple[tuple[Fraction, Fraction], ...]:
    clip = (
        clip_triangle
        if _signed_chart_area(clip_triangle) > 0
        else tuple(reversed(clip_triangle))
    )
    polygon = list(subject)
    for edge_row in range(3):
        edge_start = clip[edge_row]
        edge_end = clip[(edge_row + 1) % 3]
        direction = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
        result: list[tuple[Fraction, Fraction]] = []
        if not polygon:
            break
        start = polygon[-1]
        start_gap = _cross2(
            direction, (start[0] - edge_start[0], start[1] - edge_start[1])
        )
        for end in polygon:
            end_gap = _cross2(direction, (end[0] - edge_start[0], end[1] - edge_start[1]))
            if (start_gap >= 0) != (end_gap >= 0):
                fraction = start_gap / (start_gap - end_gap)
                intersection = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
                if not result or result[-1] != intersection:
                    result.append(intersection)
            if end_gap >= 0 and (not result or result[-1] != end):
                result.append(end)
            start, start_gap = end, end_gap
        if len(result) > 1 and result[0] == result[-1]:
            result.pop()
        polygon = result
    changed = True
    while changed and len(polygon) >= 3:
        changed = False
        for row in range(len(polygon)):
            previous = polygon[row - 1]
            current = polygon[row]
            following = polygon[(row + 1) % len(polygon)]
            if (
                _cross2(
                    (current[0] - previous[0], current[1] - previous[1]),
                    (following[0] - current[0], following[1] - current[1]),
                )
                == 0
            ):
                polygon.pop(row)
                changed = True
                break
    return tuple(polygon)


def _mapped_root_face_local_index(
    cell_kind: str,
    reference_points: tuple[tuple[Fraction, ...], ...],
    /,
) -> int:
    topology = reference_cell_topology(cell_kind)
    vertices = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(topology.vertices, dtype=np.float64)
    )
    matches = []
    for local_face, face in enumerate(topology.entities[2]):
        points = tuple(vertices[int(index)] for index in face)
        first = tuple(points[1][axis] - points[0][axis] for axis in range(3))
        second = tuple(points[2][axis] - points[0][axis] for axis in range(3))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        if all(
            sum(normal[axis] * (point[axis] - points[0][axis]) for axis in range(3)) == 0
            for point in reference_points
        ):
            matches.append(local_face)
    if len(matches) != 1:
        raise ValueError(
            "Mapped layer boundary must lie on one unique original root face."
        )
    return int(matches[0])


def _layer_target_face_orientation(
    face_order: tuple[int, ...], source_vertex_order: tuple[int, ...], /
) -> int:
    if len(face_order) != 3 or len(source_vertex_order) != 3:
        raise ValueError(
            "Mapped layer source charts require triangular target wall facets."
        )
    permutation = tuple(
        face_order.index(identifier) for identifier in source_vertex_order
    )
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _mapped_layer_surface_chart_witness(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    query: MappedReferenceBoundarySource | None = None,
    /,
) -> SourceBoundaryChartWitness:
    if query is None:
        fidelity_source = source.fidelity_source
        if not isinstance(fidelity_source, MappedReferenceBoundarySource):
            raise TypeError(
                "Mapped layer source fidelity requires MappedReferenceBoundarySource."
            )
        query = fidelity_source
    controls, nested_maps = _mapped_layer_reference_controls(source, construction, query)
    authority = query.authority
    domain = source.mapped_domain
    if domain is None:
        raise ValueError("Mapped layer source lost its original reference domain.")
    root_face_index = _mapped_layer_root_face_index(query)

    chart_rows_by_root_face: dict[
        int, list[tuple[str, int, tuple[str, tuple[str, ...]]]]
    ] = {}
    chart_ids = query.face_chart_ids
    chart_root_faces = np.asarray(query.polygon_root_face_global_ids, dtype=np.int64)
    chart_root_cells = np.asarray(query.polygon_root_cell_global_ids, dtype=np.int64)
    chart_root_local_faces = np.asarray(
        query.polygon_root_local_face_indices, dtype=np.int64
    )
    chart_source_ids = np.asarray(query.polygon_source_entity_ids, dtype=np.int64)
    chart_source_keys = query.polygon_source_entity_keys
    chart_occurrences = query.polygon_source_occurrence_paths
    if any(
        len(values) != len(chart_ids)
        for values in (
            chart_root_faces,
            chart_root_cells,
            chart_root_local_faces,
            chart_source_ids,
            chart_source_keys,
            chart_occurrences,
        )
    ):
        raise ValueError("Mapped layer boundary query chart banks are misaligned.")
    source_scope_keys = {
        source.source_entity_key_for_scope(2, int(identifier))
        for identifier in np.asarray(source.scope_entity_ids(2), dtype=np.int64).tolist()
    }
    authority_scope_keys = {
        authority.source_entity_key_for_scope(2, int(identifier))
        for identifier in np.asarray(
            authority.scope_entity_ids(2), dtype=np.int64
        ).tolist()
    }
    for row, chart_id in enumerate(chart_ids):
        source_key = query.chart_source_key(2, chart_id)
        relation_key = (chart_source_keys[row], chart_occurrences[row])
        if (
            source_key != relation_key
            or source_key not in source_scope_keys
            or source_key not in authority_scope_keys
        ):
            raise ValueError("Mapped layer chart lost its authored PLC scope identity.")
        chart_rows_by_root_face.setdefault(int(chart_root_faces[row]), []).append(
            (chart_id, row, source_key)
        )

    root_cell_regions: dict[int, int] = {}
    root_region_cursor = 0
    for block in domain.reference_mesh.blocks:
        for cell_id in np.asarray(block.global_ids, dtype=np.int64).tolist():
            root_cell_regions[int(cell_id)] = int(domain.cell_regions[root_region_cursor])
            root_region_cursor += 1
    if root_region_cursor != np.asarray(domain.cell_regions).size:
        raise ValueError(
            "Mapped layer root region table does not cover its reference cells."
        )

    wall = source.layers.source_wall
    wall_map = np.asarray(source.layers.wall_vertices, dtype=np.int64)
    mesh = construction.mesh
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if (
        wall_map.shape != np.asarray(wall.vertex_global_ids).shape
        or np.any(wall_map < 0)
        or np.any(wall_map >= target_vertex_ids.size)
    ):
        raise ValueError("Mapped layer wall-to-output vertex incidence is incomplete.")
    target_faces = _layer_output_boundary_faces(mesh)
    standard_chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    chart_cell_ids: list[int] = []
    chart_entity_ids: list[int] = []
    chart_local_faces: list[int] = []
    target_reference_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    source_parameter_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    witness_chart_ids: list[str] = []
    witness_entity_ids: list[str] = []
    witness_occurrences: list[tuple[str, ...]] = []
    witness_orientations: list[int] = []
    covered_target_faces: set[int] = set()
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Mapped layer source wall must retain triangular facets.")
        for source_vertices in np.asarray(block.vertices, dtype=np.int64):
            mapped_rows = wall_map[source_vertices]
            mapped_vertex_ids = tuple(
                int(value) for value in target_vertex_ids[mapped_rows]
            )
            candidates = target_faces.get(tuple(sorted(mapped_vertex_ids)), ())
            if len(candidates) != 1:
                raise ValueError(
                    "Each mapped layer wall triangle must bind one unique output boundary facet."
                )
            (
                target_face_id,
                target_cell_id,
                target_local_face,
                _,
                target_face_order,
                target_parameters_by_vertex,
            ) = candidates[0]
            if target_face_id in covered_target_faces:
                raise ValueError(
                    "Mapped layer source wall repeats an output boundary facet."
                )
            covered_target_faces.add(target_face_id)
            target_reference_triangle = tuple(
                target_parameters_by_vertex[identifier]
                for identifier in mapped_vertex_ids
            )
            control = controls[target_cell_id]
            if (
                root_cell_regions.get(int(control.source_cell_id))
                != control.source_region
                or control.source_region_id != domain.region_ids[control.source_region]
            ):
                raise ValueError("Mapped layer target/reference region ancestry changed.")
            nested_map = nested_maps[target_cell_id]
            root_reference_triangle = tuple(
                _exact_fraction_point(nested_map.evaluate(point))
                for point in target_reference_triangle
            )
            root_face_local = _mapped_root_face_local_index(
                control.source_cell_kind, root_reference_triangle
            )
            root_face_id = root_face_index.get(
                (int(control.source_cell_id), root_face_local)
            )
            if root_face_id is None:
                raise ValueError(
                    "Mapped layer wall is not incident to a declared source PLC root face."
                )
            target_uv = tuple(
                _exact_fraction_pair(point)
                for point in query.root_face_uv(
                    root_face_id,
                    int(control.source_cell_id),
                    root_face_local,
                    np.asarray(root_reference_triangle, dtype=object),
                )
            )
            target_area = abs(_signed_chart_area(target_uv))
            if target_area == 0:
                raise ValueError(
                    "Mapped layer wall collapses in the original root-face chart."
                )
            face_charts = chart_rows_by_root_face.get(root_face_id, ())
            if not face_charts:
                continue
            target_orientation = _layer_target_face_orientation(
                target_face_order, mapped_vertex_ids
            )
            covered_area = Fraction(0)
            for chart_id, relation_row, source_key in face_charts:
                # The query keeps one selected root-cell side per polygon.
                if int(chart_root_cells[relation_row]) != int(control.source_cell_id):
                    continue
                if int(chart_root_local_faces[relation_row]) != root_face_local:
                    raise ValueError(
                        "Mapped layer source chart changes its root local-face incidence."
                    )
                if source_key[1] != control.source_occurrence_path:
                    raise ValueError(
                        "Mapped layer root cell and source chart occurrences disagree."
                    )
                if (
                    root_cell_regions[int(control.source_cell_id)]
                    != control.source_region
                ):
                    raise ValueError(
                        "Mapped layer source chart belongs to a different material region."
                    )
                source_uv = tuple(
                    _exact_fraction_pair(point) for point in query.chart_face_uv(chart_id)
                )
                clipped = _clip_exact_triangle(target_uv, source_uv)
                if len(clipped) < 3:
                    continue
                covered_area += abs(_signed_chart_area(clipped))
                for triangle_row in range(1, len(clipped) - 1):
                    uv_triangle = (
                        clipped[0],
                        clipped[triangle_row],
                        clipped[triangle_row + 1],
                    )
                    target_reference_piece = _triangle_chart_map(
                        uv_triangle, target_uv, target_reference_triangle
                    )
                    source_parameter_piece = tuple(
                        _exact_fraction_pair(point)
                        for point in _triangle_chart_map(
                            uv_triangle, source_uv, standard_chart
                        )
                    )
                    source_orientation = (
                        1 if _signed_chart_area(source_parameter_piece) > 0 else -1
                    )
                    target_orientation_piece = target_orientation * (
                        1
                        if _signed_chart_area(uv_triangle) * _signed_chart_area(target_uv)
                        > 0
                        else -1
                    )
                    chart_cell_ids.append(int(target_cell_id))
                    chart_entity_ids.append(int(target_face_id))
                    chart_local_faces.append(int(target_local_face))
                    target_reference_rows.append(target_reference_piece)
                    source_parameter_rows.append(source_parameter_piece)
                    witness_chart_ids.append(chart_id)
                    witness_entity_ids.append(source_key[0])
                    witness_occurrences.append(source_key[1])
                    witness_orientations.append(
                        source_orientation * target_orientation_piece
                    )
            if covered_area != target_area:
                raise ValueError(
                    "Mapped source charts do not exactly partition a layer wall facet."
                )

    if not chart_cell_ids:
        raise ValueError("Mapped source chart query produced no layer wall witnesses.")
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(construction.geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(chart_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(chart_entity_ids, dtype=np.int64),
        local_entity_indices=np.asarray(chart_local_faces, dtype=np.int32),
        target_reference_vertices=np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in target_reference_rows
            ],
            dtype=np.float64,
        ),
        target_reference_rational_tokens=rational_chart_tokens(target_reference_rows),
        source_chart_ids=tuple(witness_chart_ids),
        source_entity_ids=tuple(witness_entity_ids),
        occurrence_paths=tuple(witness_occurrences),
        source_parameters=np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in source_parameter_rows
            ],
            dtype=np.float64,
        ),
        source_parameter_rational_tokens=rational_chart_tokens(source_parameter_rows),
        orientations=np.asarray(witness_orientations, dtype=np.int8),
    )
    source_fidelity_owner.validate_source_fidelity_witness(
        mesh, construction.geometry, query, witness
    )
    return witness


def _layer_feature_boundary_query(
    source: sources_owner.NativeLayerCoreSource,
    feature: ProtectedFeature,
    /,
) -> MeshingDomainBoundarySource | MappedReferenceBoundarySource:
    """Resolve a protected scope through its original source entity keys."""
    charts = source.source_surface_charts
    scope = feature.scope
    if charts is None:
        query = source.fidelity_source
        if not isinstance(query, MappedReferenceBoundarySource) or not isinstance(
            query.authority, sources_owner.NativeMappedHexSource
        ):
            raise ValueError(
                "Layer feature fidelity requires retained exact source charts."
            )
        if (
            scope.entity_dimension != 2
            or scope.source_id != source.source_id
            or scope.source_revision != source.source_revision
            or scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or scope.entity_set_id != source.scope_entity_set_id(2)
        ):
            raise ValueError("Mapped layer feature changes its source scope identity.")
        authority_keys = {
            query.authority.source_entity_key_for_scope(2, int(identifier))
            for identifier in np.asarray(
                query.authority.scope_entity_ids(2), dtype=np.int64
            ).tolist()
        }
        keys = []
        for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist():
            key = source.source_entity_key_for_scope(2, int(identifier))
            if key not in authority_keys:
                raise ValueError(
                    "Mapped layer feature source identity differs from its authority."
                )
            keys.append(key)
        return query.authority.boundary_query(face_scope_keys=tuple(keys))
    if (
        scope.entity_dimension != 2
        or scope.source_id != source.source_id
        or scope.source_revision != source.source_revision
        or scope.entity_set_id != source.scope_entity_set_id(2)
    ):
        raise ValueError("Layer surface feature changes its source scope identity.")
    domain = charts.domain
    patch_rows: set[int] = set()
    for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist():
        entity_key, occurrence = source.source_entity_key_for_scope(2, int(identifier))
        patch_rows.add(
            domain.resolve_entity_id(2, entity_key, occurrence_path=occurrence)
        )
    patches = tuple(sorted(patch_rows))
    if not patches or not set(patches) <= set(charts.boundary_source.patches):
        raise ValueError("Layer source feature is absent from the retained root charts.")
    return MeshingDomainBoundarySource(domain, patches)


def _layer_boundary_chart_witness(
    source: sources_owner.NativeLayerCoreSource,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    query: SourceBoundaryQuery,
    /,
) -> SourceBoundaryChartWitness:
    charts = source.source_surface_charts
    if charts is not None:
        if not isinstance(query, MeshingDomainBoundarySource):
            raise TypeError(
                "Layer surface charts require their retained domain boundary query."
            )
        return _layer_surface_chart_witness(source, construction, query)
    if isinstance(query, MappedReferenceBoundarySource):
        return _mapped_layer_surface_chart_witness(source, construction, query)
    if query is source.fidelity_source:
        return _implicit_layer_surface_chart_witness(source, construction)
    raise TypeError("Layer witness query has no retained exact source chart authority.")


def _native_construction(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    coordinate_contract: SpatialCoordinateContract,
    construction: layer_core_owner.layer_topology.LayerCoreConstruction,
    curving_result: layer_curving_owner.LayerCurvingConstructionResult | None = None,
    /,
) -> construction_owner.NativeMeshConstruction:
    """Lower complete route data to the canonical acceptance contract."""
    size = specification.size_controls[0]
    if not isinstance(size, sizing_owner.UniformSizeControl):
        raise TypeError("Layer/core certification requires a uniform size control.")
    if source.fidelity_source is None:
        raise ValueError("Layer certification requires its retained source query.")
    explicit_tolerances = tuple(
        value
        for value in (source.maximum_deviation, specification.maximum_deviation)
        if value is not None
    )
    if not explicit_tolerances:
        raise ValueError(
            "Layer source fidelity requires an explicit physical deviation tolerance."
        )
    fidelity_tolerance = min(explicit_tolerances)
    charts = source.source_surface_charts
    if curving_result is None:
        query = source.fidelity_source if charts is None else charts.boundary_source
        if query is None:
            raise ValueError("Layer chart witness requires its retained source query.")
        fidelity_witness = _layer_boundary_chart_witness(source, construction, query)
    else:
        if (
            curving_result.status is not layer_curving_owner.LayerCurvingStatus.CURVED
            or curving_result.chart_witness is None
            or curving_result.source_fidelity is None
        ):
            raise ValueError(
                "Accepted layer curving must retain its source chart witness and fidelity."
            )
        fidelity_witness = curving_result.chart_witness
    source_fidelity_owner.validate_source_fidelity_witness(
        construction.mesh,
        construction.geometry,
        source.fidelity_source,
        fidelity_witness,
    )
    scoped_fidelity = tuple(
        (
            query,
            _layer_boundary_chart_witness(source, construction, query),
            feature.maximum_deviation,
        )
        for feature in specification.protected_features
        if feature.hard
        and feature.feature_kind in (FeatureKind.SURFACE, FeatureKind.MATERIAL_INTERFACE)
        for query in (_layer_feature_boundary_query(source, feature),)
    )
    soft_feature_outcomes = []
    for feature in specification.protected_features:
        if feature.hard or feature.feature_kind not in (
            FeatureKind.SURFACE,
            FeatureKind.MATERIAL_INTERFACE,
        ):
            continue
        query = _layer_feature_boundary_query(source, feature)
        witness = _layer_boundary_chart_witness(source, construction, query)
        certificate = source_fidelity_owner.certify_source_fidelity(
            construction.mesh,
            construction.geometry,
            query,
            tolerance=feature.maximum_deviation,
            witness=witness,
        )
        deviation_upper = max(
            certificate.mesh_to_source_upper,
            certificate.source_to_mesh_upper,
        )
        warning = (
            certificate.status != "certified"
            or deviation_upper > feature.maximum_deviation
        )
        soft_feature_outcomes.append((feature, deviation_upper, warning))
    feature_source = source if specification.protected_features else None
    certification = construction_owner.NativeCertificationRequest(
        MeshCertificationSchedule("volume_layer"),
        source.source_id,
        source.source_revision,
        domain=construction.domain,
        cell_regions=construction.cell_regions,
        fidelity_source=source.fidelity_source,
        fidelity_tolerance=fidelity_tolerance,
        scoped_fidelity=scoped_fidelity,
        protected_features=specification.protected_features,
        fidelity_witness=fidelity_witness,
        maximum_normal_deviation=None,
        feature_source=feature_source,
    )
    organization = construction_owner.NativeMeshOrganization(
        patches=construction.patches,
        zones=construction.zones,
        labels=construction.labels,
        attributes=construction.attributes,
        associations=construction.associations,
    )
    stages = tuple(construction.core.stages)
    layer_curving_evidence = None
    if curving_result is not None:
        if curving_result.status is not layer_curving_owner.LayerCurvingStatus.CURVED:
            raise ValueError(
                "Only an accepted CURVED layer result may enter construction publication."
            )
        evidence = curving_result.source_evidence
        layer_curving_evidence = evidence
        projection = curving_result.projection_trace
        witness = curving_result.chart_witness
        fidelity = curving_result.source_fidelity
        validity = curving_result.validity
        embedding = curving_result.embedding
        if (
            evidence is None
            or projection is None
            or witness is None
            or fidelity is None
            or validity is None
            or embedding is None
        ):
            raise ValueError(
                "A CURVED layer result lost one of its retained source/validity witnesses."
            )
        stages += (
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_AUDIT,
                MeshingStageStatus.PASSED,
                input_ids=(construction.mesh.mesh_id, curving_result.result_id),
                output_ids=(
                    evidence.evidence_id,
                    projection.trace_id,
                    witness.witness_id,
                    fidelity.certificate_id,
                    validity.certificate_id,
                    embedding.certificate_id,
                ),
            ),
        )
    return construction_owner.NativeMeshConstruction(
        construction.mesh,
        coordinate_contract,
        _compliance_report(
            source,
            specification,
            construction,
            soft_feature_outcomes=tuple(soft_feature_outcomes),
        ),
        certification,
        geometry=construction.geometry,
        organization=organization,
        stages=stages,
        reference_maps=construction.reference_maps,
        source_mesh=construction.source_mesh,
        source_geometry=construction.source_geometry,
        layer_curving_evidence=layer_curving_evidence,
    )


@final
class PreparedNativeLayer(StrictModule, NonTrainableState):
    """Immutable accepted-input route data; execution creates only construction records."""

    source: sources_owner.NativeLayerCoreSource
    specification: VolumeMeshingSpec
    options: options_owner.NativeMeshingOptions
    coordinate_contract: SpatialCoordinateContract
    core_specification: VolumeMeshingSpec

    @checked
    def __init__(
        self,
        source: sources_owner.NativeLayerCoreSource,
        specification: VolumeMeshingSpec,
        options: options_owner.NativeMeshingOptions,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        if options.route != "layer_core":
            raise ValueError("NativeMeshingOptions.route must select layer_core.")
        polyhedral = specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
        if polyhedral and source.core_power_source is None:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Polyhedral layer/core fill requires an authored core power source.",
                provider_code="native_layer_core_polyhedral",
                stage="route_admission",
            )
        mapped_domain = source.mapped_domain
        if (
            mapped_domain is not None
            and mapped_domain.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Mapped layer source and target use different spatial coordinate contracts."
            )
        issues = layer_core_support_issues(source, specification)
        if issues:
            message = (
                "Native layer/core route does not support the requested semantics: "
                + "; ".join(issues)
            )
            if polyhedral:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    message,
                    provider_code="native_layer_core_polyhedral",
                    stage="route_admission",
                )
            raise ValueError(message)
        if options.volume_schedule is None:
            raise ValueError("Native layer/core execution requires its volume schedule.")
        core_specification = _lower_core_specification(source, specification)
        if polyhedral:
            core_power_source = source.core_power_source
            if core_power_source is None:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                    "Polyhedral layer/core fill requires an authored core power source.",
                    provider_code="native_layer_core_polyhedral",
                    stage="route_admission",
                )
            polyhedral_issues = polyhedral_support_issues(
                core_power_source, core_specification
            )
            if polyhedral_issues:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Native polyhedral layer/core route does not support the requested semantics: "
                    + "; ".join(polyhedral_issues),
                    provider_code="native_layer_core_polyhedral",
                    stage="route_admission",
                )
        self.source = source
        self.specification = specification
        self.options = options
        self.coordinate_contract = coordinate_contract
        self.core_specification = core_specification

    @checked
    def execute(self) -> construction_owner.NativeMeshConstruction:
        schedule = self.options.volume_schedule
        if schedule is None:
            raise ValueError("Prepared layer/core route lost its volume schedule.")
        polyhedral_schedule = self.options.polyhedral_schedule
        if (
            self.specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
            and polyhedral_schedule is None
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Native layer/core route lost its polyhedral power schedule.",
                provider_code="native_layer_core_polyhedral",
                stage="route_execution",
            )
        layer_construction = layer_core_owner.generate_layer_core(
            self.source,
            self.specification,
            self.core_specification,
            schedule,
            frame_id=self.coordinate_contract.spatial_id,
            polyhedral_schedule=polyhedral_schedule,
        )
        curving_result = None
        implicit_root_witness = None
        mapped_chart_witness = None
        if self.specification.target.geometry_order > 1:
            if self.source.source_surface_charts is None:
                query = self.source.fidelity_source
                if query is None:
                    raise ValueError(
                        "Chartless layer curving requires its original source query."
                    )
                if isinstance(query, MappedReferenceBoundarySource):
                    mapped_chart_witness = _mapped_layer_surface_chart_witness(
                        self.source, layer_construction, query
                    )
                else:
                    wall = self.source.layers.source_wall
                    implicit_root_witness = _implicit_layer_root_witness(
                        self.source.source_id,
                        self.source.source_revision,
                        query,
                        wall,
                        CellGeometrySpec.affine(wall),
                    )
            curving_result = layer_curving_owner.curve_layer_core_construction(
                layer_construction,
                self.source.layers,
                self.source.source_surface_charts,
                policy=high_order_curving_owner.HighOrderCurvingPolicy(
                    degree=self.specification.target.geometry_order,
                    relaxation_rounds=0,
                ),
                fidelity_tolerance=self.specification.maximum_deviation,
                maximum_normal_deviation=None,
                fidelity_source=(
                    self.source.fidelity_source
                    if self.source.source_surface_charts is None
                    else None
                ),
                implicit_root_witness=implicit_root_witness,
                mapped_chart_witness=mapped_chart_witness,
                frame_id=(
                    self.coordinate_contract.spatial_id
                    if self.source.source_surface_charts is None
                    else None
                ),
            )
            layer_construction = layer_curving_owner.require_layer_core_construction(
                curving_result
            )
        return _native_construction(
            self.source,
            self.specification,
            self.coordinate_contract,
            layer_construction,
            curving_result,
        )


@checked
def prepare_native_layer(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    options: options_owner.NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedNativeLayer:
    """Prepare exact layer/core source controls without allocating execution ledgers."""
    return PreparedNativeLayer(source, specification, options, coordinate_contract)


__all__ = ["PreparedNativeLayer", "layer_core_support_issues", "prepare_native_layer"]
