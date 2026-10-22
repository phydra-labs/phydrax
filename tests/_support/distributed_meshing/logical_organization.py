# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Logical organization and sparse distributed lookup scenarios."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexState,
    coarsen_adaptive_simplex_parts,
)
from phydrax.meshing._organization import validate_local_mesh_organization
from phydrax.meshing._scope import local_logical_lookup
from phydrax.meshing.providers._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import NativePublicationPolicy


M, D = phx.meshing, phx.discretization


def sparse_inventory() -> dict[str, Any]:
    # `import phydrax` initialized this rank from the runner's PHYDRAX_* variables.
    mesh = Mesh(np.asarray(jax.devices()), ("inventory",))
    sharding = NamedSharding(mesh, PartitionSpec("inventory"))
    scalar_ids = np.asarray((10, 20, 30, 40), dtype=np.int64)
    scalar_values = np.asarray((100, 200, 300, 400), dtype=np.int32)
    identifiers = jax.make_array_from_callback(
        (4,), sharding, lambda index: scalar_ids[index]
    )
    values = jax.make_array_from_callback(
        (4,), sharding, lambda index: scalar_values[index]
    )
    queries = (
        jnp.asarray((), dtype=jnp.int64)
        if jax.process_index() == 0
        else jnp.asarray((20, 99, 40), dtype=jnp.int64)
    )
    selected, valid = local_logical_lookup(identifiers, values, queries)
    membership = local_logical_lookup(identifiers, None, queries)[1]
    key_ids = np.asarray(((10, 1), (10, 2), (20, 1), (30, 4)), dtype=np.int64)
    key_sharding = NamedSharding(mesh, PartitionSpec("inventory", None))
    keys = jax.make_array_from_callback(
        (4, 2), key_sharding, lambda index: key_ids[index]
    )
    key_queries = (
        jnp.empty((0, 2), dtype=jnp.int64)
        if jax.process_index() == 0
        else jnp.asarray(((10, 2), (99, 0), (30, 4)), dtype=jnp.int64)
    )
    key_values, key_valid = local_logical_lookup(keys, values, key_queries)
    record = {
        "selected": np.asarray(selected).tolist(),
        "valid": np.asarray(valid).tolist(),
        "membership": np.asarray(membership).tolist(),
        "key_values": np.asarray(key_values).tolist(),
        "key_valid": np.asarray(key_valid).tolist(),
        "selected_dtype": str(selected.dtype),
        "valid_dtype": str(valid.dtype),
        "membership_dtype": str(membership.dtype),
        "key_values_dtype": str(key_values.dtype),
        "key_valid_dtype": str(key_valid.dtype),
    }
    jax.distributed.shutdown()
    return record


def _organization_parts() -> tuple[
    M.CellMeshingResult, M.PartitionedAdaptiveSimplex, AdaptiveSimplexState
]:
    points = np.asarray(
        ((0, 0), (1, 0), (0, 1), (3, 0), (4, 0), (3, 1)), dtype=np.float64
    )
    cells = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    seed = M.certify_cell_mesh(
        D.CellMesh.from_triangles(points, cells), phx.SpatialCoordinateContract.si()
    )
    mesh = seed.mesh
    scope = M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        mesh.entity_set(2).entity_set_id,
        np.asarray((1,), dtype=np.int64),
    )
    label = M.MeshLabel("remote-cell", scope)
    zone = M.MeshZone(
        "remote-material",
        M.MeshZoneRole.REGION,
        scope,
        material_id="water",
        region_role=M.RegionRole.FLUID,
    )
    boundary = np.asarray(
        ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)), dtype=np.int32
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        points,
        boundary,
        np.tile(np.asarray((0, -1), dtype=np.int32), (boundary.shape[0], 1)),
        ("domain",),
        source_id="disjoint-triangles",
    )
    limits = M.MeshingLimits()
    with (
        NativeExecutionScope(limits.native_limits()) as execution_scope,
        execution_scope.host_workspace() as workspace,
    ):
        source = publish_native_result(
            mesh,
            seed.coordinate_contract,
            M.MeshingComplianceReport("disjoint-triangles"),
            (),
            seed.provider,
            {"kind": "independent-domain-certified-source"},
            NativeCertificationRequest(
                M.MeshCertificationSchedule("volume_plc"),
                "disjoint-triangles",
                "r0",
                limits,
                domain=domain,
                cell_regions=np.zeros((2,), dtype=np.int32),
            ),
            policy=NativePublicationPolicy(
                M.CellMeshAuditPolicy(
                    watertight_boundary=M.CellMeshAuditDisposition.REJECT
                ),
                M.MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
                (),
                (),
            ),
            organization=NativePublicationOrganization(zones=(zone,), labels=(label,)),
            execution=NativePublicationExecution(execution_scope, workspace, None, None),
        )
    partition_policy = M.MeshPartitionPolicy(M.MeshPartitionKind.MORTON, 2, halo_width=2)
    distribution = M.prepare_mesh_distribution(
        M.MeshPart("domain", source), policy=partition_policy
    )
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=64, cell_capacity=64),
        distribution=distribution,
        partition_policy=partition_policy,
        audit_policy=M.CellMeshAuditPolicy(
            watertight_boundary=M.CellMeshAuditDisposition.REJECT
        ),
    )
    partitioned = M.partition_adaptive_simplex(
        M.prepare_adaptive_simplex(source, policy=policy)
    )
    marks = partitioned.cell_marks(np.asarray((0, 1), dtype=np.int64))
    states = D.refine_adaptive_simplex_parts(partitioned.layout, type(partitioned).execution_parts(partitioned), partitioned.states, marks).state
    return source, partitioned, states


def _inventory_record(
    source: M.CellMeshingResult, results: tuple[M.MeshAdaptationResult, ...]
) -> dict[str, Any]:
    records = []
    for result in results:
        target = result.target
        storage = target.mesh.storage
        if storage is None:
            raise AssertionError(
                "organization_fixture.storage must retain logical storage; got None."
            )
        scope = target.labels[0].scope
        selected = M.resolve_mesh_scope(target.mesh, scope)
        union = scope.union(scope)
        records.append(
            {
                "global_label_ids": np.asarray(scope.global_entity_ids).tolist(),
                "global_zone_ids": np.asarray(
                    target.zones[0].scope.global_entity_ids
                ).tolist(),
                "material_id": target.zones[0].material_id,
                "region_role": target.zones[0].region_role.value
                if target.zones[0].region_role is not None
                else None,
                "selected_ids": np.sort(
                    np.asarray(target.mesh.entity_set(2).entity_ids)[
                        np.asarray(selected.mask)
                    ]
                ).tolist(),
                "local_ids": np.asarray(scope.entity_ids).tolist(),
                "coverage_id": scope.local_coverage_id,
                "storage_id": storage.evidence_id,
                "label_id": target.labels[0].label_id,
                "scope_id": scope.scope_id,
                "union_scope_id": union.scope_id,
                "union_ids": np.asarray(union.entity_ids).tolist(),
            }
        )
    target = results[1].target
    storage = target.mesh.storage
    if storage is None:
        raise AssertionError(
            "organization_fixture.storage must retain logical storage; got None."
        )
    arrays = dict(storage.logical_arrays)

    def rejected(
        patches: tuple[M.MeshPatch, ...],
        zones: tuple[M.MeshZone, ...],
        labels: tuple[M.MeshLabel, ...],
        attributes: tuple[M.MeshAttribute, ...],
    ) -> bool:
        try:
            validate_local_mesh_organization(
                source, target.mesh, arrays, patches, zones, labels, attributes
            )
        except ValueError:
            return True
        return False

    altered = M.MeshZone(
        target.zones[0].name,
        target.zones[0].role,
        target.zones[0].scope,
        material_id="oil",
        region_role=M.RegionRole.FLUID,
    )
    wrong_scope = eqx.tree_at(
        lambda value: value.entity_owner,
        target.labels[0].scope,
        jnp.zeros(target.labels[0].scope.entity_owner.shape, dtype=jnp.int32),
    )
    wrong_label = eqx.tree_at(lambda value: value.scope, target.labels[0], wrong_scope)
    overlap_rejected = False
    empty_scope = results[0].target.zones[0].scope
    try:
        M.validate_mesh_zones(
            (
                M.MeshZone("first", M.MeshZoneRole.BOUNDARY, empty_scope),
                M.MeshZone("second", M.MeshZoneRole.BOUNDARY, empty_scope),
            )
        )
    except ValueError:
        overlap_rejected = True
    patches = tuple(target.patches)
    return {
        "locals": records,
        "material_change_rejected": rejected(
            patches, (altered,), target.labels, target.attributes
        ),
        "dropped_label_rejected": rejected(patches, target.zones, (), target.attributes),
        "wrong_owner_rejected": rejected(
            patches, target.zones, (wrong_label,), target.attributes
        ),
        "global_overlap_rejected": overlap_rejected,
    }


def refine_organization() -> dict[str, Any]:
    source, partitioned, states = _organization_parts()
    results = M.commit_partitioned_adaptive_simplex(partitioned, states).local_results
    return _inventory_record(source, results)


def coarsen_organization() -> dict[str, Any]:
    source, partitioned, states = _organization_parts()
    states = coarsen_adaptive_simplex_parts(partitioned.layout, type(partitioned).execution_parts(partitioned), states, states.mesh.cell_active).state
    results = M.commit_partitioned_adaptive_simplex(partitioned, states).local_results
    return _inventory_record(source, results)
