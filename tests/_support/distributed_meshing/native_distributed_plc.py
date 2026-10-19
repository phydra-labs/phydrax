# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from fractions import Fraction
from typing import Any

import jax
import numpy as np

import phydrax as phx
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.meshing._association import PlcAssociationTransfer
from phydrax.meshing.providers._native_planar_domain import source_arrays
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
    _require_instance,
    _require_present,
)


_CONTRACT_KEYS = (
    "plc_30",
    "plc_48",
    "plc_50",
    "plc_53",
    "plc_54",
    "plc_58",
    "plc_59",
    "plc_60",
    "plc_61",
    "plc_62",
    "plc_63",
    "plc_65",
    "plc_67",
    "plc_80",
    "plc_81",
    "plc_90",
    "plc_92",
    "plc_94",
)


def planar_plc_transfer() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    # `import phydrax` initialized this rank from the runner's PHYDRAX_* variables.
    M = phx.meshing
    D = phx.discretization
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(
            [
                [0, 0],
                [2, 0],
                [2, 2],
                [0, 2],
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [1.5, 0.5],
            ],
            dtype=np.float64,
        ),
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        feature_id="plate",
    )
    authored = M.NativePlanarSource(region, "r1")
    scope = M.MeshingScope(
        "plate", "r1", M.MeshingEntityKind.GEOMETRY, 2, "plate-region", np.asarray([0])
    )
    source = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("planar_constrained_delaunay"))
        .plan(
            authored,
            M.SurfaceMeshingSpec(
                M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
                scope,
                planar_embedding=phx.geometry.PlanarEmbedding(
                    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
                ),
                size_controls=(
                    M.UniformSizeControl(
                        scope, 1.0, maximum_size=1.5, strength=M.SizeControlStrength.HARD
                    ),
                ),
                size_compliance=M.SizeCompliancePolicy(
                    relative_tolerance=0.5, target_statistics=("p50",)
                ),
            ),
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    checks.setdefault("plc_30", []).append(
        bool(
            source.certification is not None and source.certification.coverage is not None
        )
    )
    certification = _require_present(source.certification)
    domain = _require_instance(
        certification.request.domain, phx.geometry.PiecewiseLinearDomain
    )
    _, edges, _ = source_arrays(authored)
    transfer = PlcAssociationTransfer(
        domain,
        source.coordinate_contract,
        authored.source_revision,
        edge_vertices=edges,
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=domain.facet_regions,
    )
    partition_policy = M.MeshPartitionPolicy(M.MeshPartitionKind.MORTON, 2, halo_width=2)
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=128, cell_capacity=128),
        distribution=M.prepare_mesh_distribution(
            M.MeshPart("domain", source), policy=partition_policy
        ),
        partition_policy=partition_policy,
        association_transfer=transfer,
        audit_policy=M.CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=M.CellMeshAuditDisposition.REJECT,
        ),
    )
    partitioned = M.partition_adaptive_simplex(
        M.prepare_adaptive_simplex(source, policy=policy)
    )
    update = D.refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        partitioned.states.mesh.cell_active,
    )
    checks.setdefault("plc_48", []).append(
        bool(np.all(np.asarray(update.report.status.addressable_shards[0].data) == 0))
    )
    (result,) = M.commit_partitioned_adaptive_simplex(
        partitioned, update.state
    ).local_results
    checks.setdefault("plc_50", []).append(
        bool(result.status is M.MeshAdaptationStatus.COMPLETE)
    )
    target = result.target
    storage = _require_present(target.mesh.storage)
    checks.setdefault("plc_53", []).append(
        bool(storage is not None and storage.partition_index == jax.process_index())
    )
    checks.setdefault("plc_54", []).append(
        bool(target.collective_certificates is not None)
    )
    embedding, coverage = _require_present(target.collective_certificates)
    embedding.binding.require(target.mesh, target.geometry)
    coverage.binding.require(target.mesh, target.geometry)
    checks.setdefault("plc_58", []).append(
        bool(embedding.status == "certified" and (not embedding.findings))
    )
    checks.setdefault("plc_59", []).append(
        bool(coverage.status == "certified" and (not coverage.findings))
    )
    checks.setdefault("plc_60", []).append(
        bool(coverage.embedding_certificate_id == embedding.certificate_id)
    )
    checks.setdefault("plc_61", []).append(bool(coverage.domain_id == domain.domain_id))
    checks.setdefault("plc_62", []).append(
        bool(
            coverage.premise_certificate_ids
            == (
                _require_present(certification.coverage).certificate_id,
                storage.evidence_id,
            )
        )
    )
    checks.setdefault("plc_63", []).append(
        bool(coverage.covered_source_facet_count == coverage.source_facet_count)
    )
    checks.setdefault("plc_65", []).append(
        bool(
            coverage.requested_region_measures
            == coverage.achieved_region_measures
            == (3.0,)
        )
    )
    charged, required = (
        coverage.source_expression_work_units,
        coverage.source_expression_required_work_units,
    )
    checks.setdefault("plc_67", []).append(
        bool(charged is not None and required is not None and (required >= charged))
    )
    points = [
        tuple((Fraction(float(value)) for value in row))
        for row in np.asarray(target.mesh.coordinates)
    ]
    authority = [
        tuple((Fraction(float(value)) for value in row))
        for row in transfer.source_vertices
    ]

    def on_segment(point: tuple[Fraction, ...], index: int) -> bool:
        (ax, ay), (bx, by) = (
            authority[vertex] for vertex in transfer.edge_vertices[index]
        )
        return (
            (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax) == 0
            and min(ax, bx) <= point[0] <= max(ax, bx)
            and (min(ay, by) <= point[1] <= max(ay, by))
        )

    strata = {}
    for association in target.associations:
        dimension = next(
            (
                d
                for d in range(3)
                if association.target_entity_set_id
                == target.mesh.entity_set(d).entity_set_id
            )
        )
        association.validate_target(target.mesh.entity_set(dimension))
        ids = np.asarray(association.target_global_ids, dtype=np.int64)
        checks.setdefault("plc_80", []).append(
            bool(association.exact and association.complete)
        )
        _record_array(
            checks,
            "plc_81",
            np.sort(ids),
            np.sort(np.asarray(target.mesh.entity_set(dimension).entity_ids)),
        )
        kinds = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        if dimension == 0:
            rows = np.searchsorted(np.asarray(target.mesh.vertex_global_ids), ids)
            for row, kind, index in zip(
                rows.tolist(), kinds.tolist(), indices.tolist(), strict=True
            ):
                point = points[row]
                match kind:
                    case 0:
                        checks.setdefault("plc_90", []).append(
                            bool(
                                point
                                == authority[
                                    int(
                                        np.flatnonzero(transfer.vertex_indices == index)[
                                            0
                                        ]
                                    )
                                ]
                            )
                        )
                    case 1:
                        checks.setdefault("plc_92", []).append(
                            bool(
                                on_segment(
                                    point,
                                    int(
                                        np.flatnonzero(transfer.edge_indices == index)[0]
                                    ),
                                )
                            )
                        )
                    case 2:
                        checks.setdefault("plc_94", []).append(
                            bool(
                                not any(
                                    (
                                        on_segment(point, edge)
                                        for edge in range(transfer.edge_vertices.shape[0])
                                    )
                                )
                            )
                        )
                    case _:
                        raise AssertionError(
                            f"planar_associations.source_stratum must be a supported source dimension; got {kind}."
                        )
        strata[str(dimension)] = {
            str(identifier): [kind, index]
            for identifier, kind, index in zip(
                ids.tolist(), kinds.tolist(), indices.tolist(), strict=True
            )
        }
    # The certificate is local even though storage also retains global logical
    # banks. The reuse key must not materialize those non-addressable banks.
    with coordinate_reference_budget(
        queries_upper=cell_geometry_preparation_query_upper(target.geometry)
    ) as budget:
        local_validity = D.certify_cell_geometry_validity(
            target.geometry, mesh=target.mesh, budget=budget
        )
        charged_work = budget.work_units
        repeated_validity = D.certify_cell_geometry_validity(
            target.geometry, mesh=target.mesh, budget=budget
        )
        validity_reused = repeated_validity is local_validity
        validity_reuse_work = budget.work_units - charged_work
    report = {
        "partition": storage.partition_index,
        "counts": list(storage.global_entity_counts),
        "strata": strata,
        "owner_local_validity": {
            "global_storage_present": target.mesh.storage is not None,
            "nonaddressable_global_banks": any(
                not value.is_fully_addressable for _, value in storage.logical_arrays
            ),
            "all_certified": bool(local_validity.all_certified),
            "cache_reused": validity_reused,
            "reuse_work_units": validity_reuse_work,
        },
    }
    report["checks"] = checks
    jax.distributed.shutdown()
    return report
