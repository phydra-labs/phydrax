"""Consumer receipts for authenticated collective restoration and explicit replay."""

from dataclasses import fields
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._fingerprint import canonical_fingerprint, tree_content_equal
from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax._model._structure import restore_registered_fields
from phydrax.discretization._adaptive_simplex import refine_adaptive_simplex_parts
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._partition import CellPartition
from phydrax.lifecycle._meshing_source_records import (
    reconstruct_stored_fields,
    stored_record_content_id,
)
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    reconstruct_meshing_source,
    register_meshing_source_artifacts,
    write_meshing_source_closure,
)
from phydrax.meshing._device_adaptation import validate_partitioned_mesh_evidence
from phydrax.meshing._result import CellMeshingResult, CollectiveMeshEvidence
from tests._support.distributed_meshing.native_distributed import _prepare_source
from tests._support.distributed_meshing.native_distributed_records import (
    _require_instance,
    _require_present,
)


def _prior_result_id(target: CellMeshingResult, evidence: CollectiveMeshEvidence) -> str:
    """Record the pre-storage-anchor result framing for the identity ledger."""
    certificates = target.collective_certificates
    return canonical_fingerprint(
        {
            "kind": "owner-local-cell-meshing-result",
            "mesh": target.mesh.mesh_id,
            "topology": target.mesh.topology_id,
            "geometry": cell_geometry_id(target.geometry),
            "geometry_layout": target.geometry.geometry_layout_id,
            "coordinate_contract": target.coordinate_contract.spatial_id,
            "source_mesh": evidence.source_mesh_id,
            "source_audit": evidence.source_audit_id,
            "source_evidence": evidence.source_evidence_id,
            "organization": evidence.global_organization_id,
            "collective": evidence.evidence_id,
            "provider": target.provider.provider_id,
            "derivative_mode": target.derivative_mode.value,
            "collective_certificates": None
            if certificates is None
            else tuple(value.certificate_id for value in certificates),
            "surface_source": None
            if target.surface_source is None
            else target.surface_source.source_chart_id,
        }
    )


def _collective_target() -> tuple[CellMeshingResult, CellMeshingResult]:
    """Build a published tetrahedral epoch with retained uniform roots."""
    source = _prepare_source("packed_inverse_tetra", {}).source
    fine = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
        )
    )
    hierarchy = _require_instance(fine.hierarchy, phx.meshing.BisectionHierarchy)
    uniform = _require_present(hierarchy.uniform_refinement)
    fine_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in fine.target.mesh.blocks)
    )
    owners = np.where(np.isin(fine_ids, np.asarray(uniform.child_ids)[0]), 0, 1).astype(
        np.int32
    )
    partition_policy = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.PROVIDER, 2, halo_width=2, maximum_imbalance=2.0
    )
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", fine.target),
        policy=partition_policy,
        ownership=owners,
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=64, cell_capacity=64
        ),
        distribution=distribution,
        partition_policy=partition_policy,
    )
    prepared = phx.meshing.prepare_adaptive_simplex(
        fine.target, policy=policy, hierarchy=hierarchy
    )
    partitioned = phx.meshing.partition_adaptive_simplex(prepared)
    refined = refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        partitioned.cell_marks(np.asarray(uniform.child_ids)[0, :1]),
    ).state
    target = (
        phx.meshing.commit_partitioned_adaptive_simplex(partitioned, refined)
        .local_results[0]
        .target
    )
    return source, target


def _published_distribution(target: CellMeshingResult) -> phx.meshing.MeshDistribution:
    """Build the storage-bound distribution for an accepted local carrier."""
    return phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", target),
        policy=phx.meshing.MeshPartitionPolicy(
            phx.meshing.MeshPartitionKind.PROVIDER,
            2,
            halo_width=2,
            maximum_imbalance=2.0,
        ),
    )


def collective_restore(archive_path: str) -> dict[str, Any]:
    """Exercise storage authentication independently of mathematical replay."""
    source, target = _collective_target()
    evidence = _require_instance(target.collective_evidence, CollectiveMeshEvidence)
    register_meshing_source_artifacts()
    storage = _require_present(target.mesh.storage)
    partition = CellPartition(
        storage.entity_owner[-1], storage.partition_count, storage=storage
    )
    distribution = _published_distribution(target)
    with NativeExecutionScope(NativeExecutionLimits(1, 0, 0, 1 << 20, 120.0)) as scope:
        with scope.batch(1) as ticket:
            ticket.consume(1)
            with ticket.native_target():
                restored = CollectiveMeshEvidence.from_stored_fields(evidence)
                domain = _require_instance(
                    _require_present(source.certification).request.domain,
                    phx.geometry.PiecewiseLinearDomain,
                )
                restored_domain = phx.geometry.PiecewiseLinearDomain.from_stored_fields(
                    domain
                )
                restored_partition = reconstruct_meshing_source(
                    partition, meshes=(target.mesh,)
                )
                restored_distribution = reconstruct_stored_fields(distribution)
            remaining_work = ticket.consumed_work_units
    restore_usage = _require_present(scope.evidence)
    with pytest.raises(ValueError, match=r"CellPartition\.storage must retain"):
        reconstruct_meshing_source(partition)
    altered_distribution = eqx.tree_at(
        lambda value: value.cell_global_ids,
        distribution,
        distribution.cell_global_ids + 1,
    )
    with pytest.raises(
        ValueError, match=r"stored_fields\.MeshDistribution.*cell_global_ids"
    ):
        reconstruct_stored_fields(altered_distribution)
    faults: dict[str, bool] = {}
    for name, altered in (
        (
            "proof",
            eqx.tree_at(
                lambda value: value.partition_checks, evidence, ~evidence.partition_checks
            ),
        ),
        (
            "bank",
            eqx.tree_at(
                lambda value: value.logical_arrays[0][1],
                evidence,
                evidence.logical_arrays[0][1] + 1,
            ),
        ),
    ):
        with pytest.raises(
            ValueError, match=r"CollectiveMeshEvidence\.stored_content_id must match"
        ):
            CollectiveMeshEvidence.from_stored_fields(altered)
        faults[name] = True
    stored = {member.name: getattr(evidence, member.name) for member in fields(evidence)}
    stored["evidence_id"] = "tampered-proof-id"
    altered_id = restore_registered_fields(CollectiveMeshEvidence, stored)
    with pytest.raises(
        ValueError, match=r"CollectiveMeshEvidence\.stored_content_id must match"
    ):
        CollectiveMeshEvidence.from_stored_fields(altered_id)
    faults["id"] = True
    forged = eqx.tree_at(
        lambda value: value.source_exterior, evidence, ~evidence.source_exterior
    )
    forged_fields = {
        member.name: getattr(forged, member.name) for member in fields(forged)
    }
    forged_fields["stored_content_id"] = stored_record_content_id(
        forged, excluded="stored_content_id"
    )
    forged = restore_registered_fields(CollectiveMeshEvidence, forged_fields)
    # A matching internal digest is insufficient: the accepted result anchors it.
    forged_target = eqx.tree_at(lambda value: value.collective_evidence, target, forged)
    with pytest.raises(
        ValueError, match=r"CellMeshingResult\.stored_content_id must match"
    ):
        CellMeshingResult.from_stored_fields(forged_target)
    faults["anchored_forgery"] = True
    with NativeExecutionScope(
        evidence.preparation.policy.limits.native_limits()
    ) as replay_scope:
        validate_partitioned_mesh_evidence(
            _require_present(target.mesh.storage), restored
        )
    replay_usage = _require_present(replay_scope.evidence)
    limits = ArrayArchiveLimits()
    receipt = write_meshing_source_closure(
        Path(archive_path), (target, domain), limits=limits
    )
    loaded = read_meshing_source_closure(
        Path(archive_path), expected_content_id=receipt.content_id, limits=limits
    )
    return {
        "equal": tree_content_equal(evidence, restored),
        "domain_equal": tree_content_equal(domain, restored_domain),
        "partition_equal": tree_content_equal(partition, restored_partition),
        "partition_missing_storage_refused": True,
        "distribution_equal": tree_content_equal(distribution, restored_distribution),
        "distribution_tampered_bank_refused": True,
        "reservation_work": remaining_work,
        "restore_native_work": restore_usage.work_counters.construction_work
        - restore_usage.externally_charged_work,
        "restore_native_queries": restore_usage.native_primitive_queries,
        "restore_external_work": restore_usage.externally_charged_work,
        "faults": faults,
        "replay_native_work": replay_usage.work_counters.construction_work
        - replay_usage.externally_charged_work,
        "replay_native_queries": replay_usage.native_primitive_queries,
        "replay_external_work": replay_usage.externally_charged_work,
        "roundtrip": tree_content_equal((target, domain), loaded),
        "archive_content_id": receipt.content_id,
        "prior_result_id": _prior_result_id(target, evidence),
        "serial_source_id": source.result_id,
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "stored_content_id": evidence.stored_content_id,
        "parent_stored_content_id": target.stored_content_id,
    }
