#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Publication of an accepted source-realized coordinate-order epoch."""

from __future__ import annotations

import jax
import numpy as np

from ..discretization._cell_geometry_validity import cell_geometry_id
from ._adaptation import (
    _RouteOutcome,
    GeometryRealizationMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationStatus,
    PreparedMeshAdaptation,
)
from ._canonical import certify_cell_mesh
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
)
from ._organization import RegionMeshingEvidence
from ._result import CellMeshingResult


def validate_geometry_realization_adaptation(
    source: CellMeshingResult,
    request: GeometryRealizationMeshAdaptation,
    policy: MeshAdaptationPolicy,
    /,
) -> None:
    if request.source_result_id != source.result_id:
        raise ValueError("Geometry realization is stale for this exact source result.")
    correspondence = request.geometry_realization
    transition = correspondence.transition
    if transition.policy_id != policy.geometry_transition.policy_id:
        raise ValueError(
            "Geometry realization must use the exact authored transition policy."
        )
    if transition.source_geometry_id != cell_geometry_id(source.geometry):
        raise ValueError(
            "Geometry realization refers to a different source coordinate map."
        )
    if policy.association_transfer is not None:
        raise ValueError(
            "An accepted source realization already owns its source association/projection evidence."
        )
    request.source_fidelity.binding.require(source.mesh, source.geometry)
    correspondence.source_embedding.binding.require(source.mesh, source.geometry)
    correspondence.target_embedding.binding.require(
        source.mesh, request.realization.geometry
    )
    if (
        correspondence.source_embedding.status != "certified"
        or correspondence.target_embedding.status != "certified"
    ):
        raise ValueError("Source realization requires both actual physical embeddings.")
    limits = policy.limits
    certificate_limits = request.certificate_limits
    work_upper = (
        4 * certificate_limits.maximum_work_units + transition.evidence.evaluation_count
    )
    scratch_upper = max(
        certificate_limits.maximum_scratch_bytes, correspondence.peak_expression_bytes
    )
    data_bytes = sum(
        value.size * value.dtype.itemsize
        for value in jax.tree.leaves(request)
        if isinstance(value, jax.Array)
    )
    geometry_entries = sum(
        value.size for value in request.realization.geometry.resolve(source.mesh)[1]
    )
    observations = (
        ("work_units", work_upper, limits.maximum_work_units),
        ("scratch_bytes", scratch_upper, limits.maximum_scratch_bytes),
        ("data_bytes", data_bytes, limits.maximum_data_bytes),
        (
            "geometry_queries",
            2 * certificate_limits.maximum_distance_evaluations,
            limits.maximum_geometry_queries,
        ),
        ("vertices", source.audit.vertex_count, limits.maximum_vertices),
        ("edges", source.audit.entity_counts[1], limits.maximum_edges),
        (
            "faces",
            source.audit.entity_counts[2] if len(source.audit.entity_counts) > 2 else 0,
            limits.maximum_faces,
        ),
        ("cells", source.audit.entity_counts[-1], limits.maximum_cells),
        (
            "connectivity_entries",
            source.audit.connectivity_entries + geometry_entries,
            limits.maximum_connectivity_entries,
        ),
    )
    for name, upper, maximum in observations:
        if upper > maximum:
            raise ValueError(
                f"Geometry realization exceeds its declared maximum_{name}: {upper} > {maximum}."
            )
    if source.region_evidence is not None:
        coverage = request.realization.evidence.coverage
        if (
            coverage is None
            or coverage.status != "certified"
            or coverage.domain_id != source.region_evidence.domain.domain_id
        ):
            raise ValueError(
                "Source realization requires actual renewed coverage of every material region/interface."
            )


def _realized_region_evidence(
    source: CellMeshingResult, request: GeometryRealizationMeshAdaptation, /
) -> RegionMeshingEvidence | None:
    old = source.region_evidence
    if old is None:
        return None
    coverage = request.realization.evidence.coverage
    if coverage is None:
        raise ValueError(
            "The realized material domain lacks its actual coverage certificate."
        )
    return RegionMeshingEvidence(
        source.mesh,
        request.realization.geometry,
        old.source_revision,
        old.source_complex_id,
        old.cell_region_ids,
        old.region_zone_ids,
        old.interface_patch_ids,
        old.interface_definitions,
        old.interface_facets,
        old.domain,
        coverage,
    )


def execute_geometry_realization_route(
    prepared: PreparedMeshAdaptation, /
) -> _RouteOutcome:
    request = prepared.request
    if not isinstance(request, GeometryRealizationMeshAdaptation):
        raise TypeError(
            "Geometry realization execution requires its typed source-realized request."
        )
    source = prepared.source
    validate_geometry_realization_adaptation(source, request, prepared.policy)
    entities = []
    for old_set in source.mesh.topology.entity_sets:
        ids = np.asarray(old_set.entity_ids, dtype=np.int64)
        entities.append(
            EntityLineage(
                old_set.intrinsic_dimension,
                old_set.entity_set_id,
                old_set.entity_set_id,
                ids,
                ids,
                np.full(ids.shape, EntityLineageKind.PRESERVED, dtype=np.int32),
            )
        )
    lineage = MeshLineage(
        source.mesh.topology_id, source.mesh.topology_id, tuple(entities)
    )
    target = certify_cell_mesh(
        source.mesh,
        source.coordinate_contract,
        geometry=request.realization.geometry,
        audit_policy=prepared.policy.audit_policy,
        patches=source.patches,
        zones=source.zones,
        labels=source.labels,
        attributes=source.attributes,
        associations=source.associations,
        region_evidence=_realized_region_evidence(source, request),
        region_boundary_evidence=source.region_boundary_evidence,
    )
    transition = CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        MeshTransitionKind.GEOMETRY_REALIZATION,
        geometry_transition=request.geometry_realization.transition,
    )
    return _RouteOutcome(
        MeshAdaptationStatus.COMPLETE,
        target,
        transition,
        lineage,
        None,
        None,
        None,
        request.realization.evidence,
        None,
    )
