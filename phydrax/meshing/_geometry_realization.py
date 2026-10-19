#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Publication of an accepted source-realized coordinate-order epoch."""

from __future__ import annotations

from typing import assert_never

import jax
import numpy as np

from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._approximate_source_coverage import ApproximateSourceCoverageCertificate
from ..geometry._domain_coverage import DomainCoverageCertificate
from ..typing import checked
from . import _adaptation as adaptation_owner
from ._canonical import certify_cell_mesh
from ._lineage import CellMeshTransition
from ._lineage_records import (
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
)
from ._organization_records import RegionMeshingEvidence
from ._result import CellMeshingResult


@checked
def validate_geometry_realization_adaptation(
    source: CellMeshingResult,
    request: adaptation_owner.GeometryRealizationMeshAdaptation,
    policy: adaptation_owner.MeshAdaptationPolicy,
    /,
) -> None:
    if request.source_result_id != source.result_id:
        raise ValueError(
            "request.source_result_id must match source.result_id; "
            f"got {request.source_result_id!r}"
        )
    correspondence = request.geometry_realization
    transition = correspondence.transition
    if transition.policy_id != policy.geometry_transition.policy_id:
        raise ValueError(
            "geometry_realization.transition.policy_id must match policy.geometry_transition.policy_id; "
            f"got {transition.policy_id!r}"
        )
    if transition.source_geometry_id != cell_geometry_id(source.geometry):
        raise ValueError(
            "geometry_realization.transition.source_geometry_id must match source.geometry; "
            f"got {transition.source_geometry_id!r}"
        )
    if policy.association_transfer is not None:
        raise ValueError(
            "policy.association_transfer must be None for accepted source realization; "
            f"got {type(policy.association_transfer).__name__}"
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
        raise ValueError(
            "geometry_realization.source_embedding.status must be certified along with target_embedding.status; "
            f"got {correspondence.source_embedding.status!r}, {correspondence.target_embedding.status!r}"
        )
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
                f"geometry_realization.{name} must not exceed limits.maximum_{name}={maximum}; got {upper}"
            )
    if source.region_evidence is not None:
        match request.realization.evidence.coverage:
            case DomainCoverageCertificate() as coverage:
                if (
                    coverage.status != "certified"
                    or coverage.domain_id != source.region_evidence.domain.domain_id
                ):
                    raise ValueError(
                        "realization.evidence.coverage must certify source.region_evidence.domain; "
                        f"got status={coverage.status!r}, domain_id={coverage.domain_id!r}"
                    )
            case ApproximateSourceCoverageCertificate():
                raise ValueError(
                    "GeometryRealizationMeshAdaptation.coverage must provide "
                    "DomainCoverageCertificate evidence for material regions/interfaces; "
                    "got ApproximateSourceCoverageCertificate."
                )
            case None:
                raise ValueError(
                    "realization.evidence.coverage must certify source.region_evidence.domain; got None"
                )
            case invalid:
                assert_never(invalid)


def _realized_region_evidence(
    source: CellMeshingResult,
    request: adaptation_owner.GeometryRealizationMeshAdaptation,
    /,
) -> RegionMeshingEvidence | None:
    old = source.region_evidence
    if old is None:
        return None
    match request.realization.evidence.coverage:
        case DomainCoverageCertificate() as coverage:
            pass
        case ApproximateSourceCoverageCertificate():
            raise ValueError(
                "GeometryRealizationMeshAdaptation.coverage must provide "
                "DomainCoverageCertificate evidence for material regions/interfaces; "
                "got ApproximateSourceCoverageCertificate."
            )
        case None:
            raise ValueError(
                "realization.evidence.coverage must certify the realized material domain; got None"
            )
        case invalid:
            assert_never(invalid)
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
        certificate_limits=request.certificate_limits,
    )


@checked
def execute_geometry_realization_route(
    prepared: adaptation_owner.PreparedMeshAdaptation, /
) -> adaptation_owner.RouteOutcome:
    match prepared.request:
        case adaptation_owner.GeometryRealizationMeshAdaptation() as request:
            pass
        case invalid:
            raise TypeError(
                "prepared.request must be GeometryRealizationMeshAdaptation; "
                f"got {type(invalid).__name__}"
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
    return adaptation_owner.RouteOutcome(
        adaptation_owner.MeshAdaptationStatus.COMPLETE,
        target,
        transition,
        lineage,
        None,
        None,
        None,
        request.realization.evidence,
        None,
    )
