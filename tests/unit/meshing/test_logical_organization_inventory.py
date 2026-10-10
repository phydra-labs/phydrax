#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import phydrax as phx


@pytest.mark.parametrize("operation", ("refine", "coarsen"))
def test_logical_organization_preserves_remote_material_and_rejects_incomplete_views(
    operation: str,
) -> None:
    environment = dict(os.environ)
    environment.update(
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        PYTHONPATH=str(Path(phx.__file__).resolve().parents[1]),
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SCENARIO, operation],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode:
        pytest.fail(f"{completed.stdout}\n{completed.stderr}", pytrace=False)


def test_scope_lineage_rejects_ambiguous_coarsening_and_global_removal() -> None:
    from phydrax.meshing._lineage import inherit_scope

    meshing = phx.meshing
    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    coarse = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    refined = phx.discretization.CellMesh.from_triangles(
        np.concatenate((points, np.asarray(((0.5, 0.5),), dtype=np.float64))),
        np.asarray(((0, 1, 4), (1, 2, 4), (0, 4, 3), (4, 2, 3)), dtype=np.int32),
    )
    original = meshing.MeshingScope(
        coarse.mesh_id,
        coarse.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        coarse.entity_set(2).entity_set_id,
        np.asarray((1,), dtype=np.int64),
    )
    split_record = meshing.EntityLineage(
        2,
        coarse.entity_set(2).entity_set_id,
        refined.entity_set(2).entity_set_id,
        np.asarray((0, 0, 1, 1), dtype=np.int64),
        np.arange(4, dtype=np.int64),
        np.full((4,), meshing.EntityLineageKind.REFINED_FROM, dtype=np.int32),
    )
    split = meshing.MeshLineage(coarse.topology_id, refined.topology_id, (split_record,))
    members = inherit_scope(original, split, refined, "region")
    np.testing.assert_array_equal(members.global_entity_ids, (2, 3))
    merge_record = meshing.EntityLineage(
        2,
        refined.entity_set(2).entity_set_id,
        coarse.entity_set(2).entity_set_id,
        np.arange(4, dtype=np.int64),
        np.asarray((0, 0, 1, 1), dtype=np.int64),
        np.full((4,), meshing.EntityLineageKind.COARSENED_INTO, dtype=np.int32),
    )
    merge = meshing.MeshLineage(refined.topology_id, coarse.topology_id, (merge_record,))
    restored = inherit_scope(members, merge, coarse, "region")
    np.testing.assert_array_equal(restored.global_entity_ids, original.global_entity_ids)
    partial = meshing.MeshingScope(
        refined.mesh_id,
        refined.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        refined.entity_set(2).entity_set_id,
        np.asarray((2,), dtype=np.int64),
    )
    with pytest.raises(ValueError):
        inherit_scope(partial, merge, coarse, "region")
    removed_record = meshing.EntityLineage(
        2,
        coarse.entity_set(2).entity_set_id,
        refined.entity_set(2).entity_set_id,
        np.asarray((0, 0), dtype=np.int64),
        np.asarray((0, 1), dtype=np.int64),
        np.full((2,), meshing.EntityLineageKind.REFINED_FROM, dtype=np.int32),
        deleted_source_ids=np.asarray((1,), dtype=np.int64),
    )
    removed = meshing.MeshLineage(
        coarse.topology_id, refined.topology_id, (removed_record,)
    )
    with pytest.raises(ValueError):
        inherit_scope(original, removed, refined, "region")


def test_scope_lineage_collapsed_membership_uses_deciding_source_precedence() -> None:
    from phydrax.meshing._lineage import inherit_scope

    meshing = phx.meshing
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    source = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    target = phx.discretization.CellMesh.from_triangles(
        points[:3], np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    source_set, target_set = source.entity_set(2), target.entity_set(2)
    dominant = meshing.MeshingScope(
        source.mesh_id,
        source.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        source_set.entity_set_id,
        np.asarray((0,), dtype=np.int64),
    )
    collapsed = meshing.MeshingScope(
        source.mesh_id,
        source.numeric_version,
        meshing.MeshingEntityKind.MESH,
        2,
        source_set.entity_set_id,
        np.asarray((1,), dtype=np.int64),
    )
    record = meshing.EntityLineage(
        2,
        source_set.entity_set_id,
        target_set.entity_set_id,
        np.asarray((1, 0), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int64),
        np.asarray(
            (
                meshing.EntityLineageKind.COLLAPSED_INTO,
                meshing.EntityLineageKind.PRESERVED,
            ),
            dtype=np.int32,
        ),
    )
    lineage = meshing.MeshLineage(source.topology_id, target.topology_id, (record,))
    inherited = inherit_scope(dominant, lineage, target, "material")
    np.testing.assert_array_equal(inherited.global_entity_ids, (0,))
    with pytest.raises(ValueError):
        inherit_scope(collapsed, lineage, target, "material")
    only_collapsed = meshing.EntityLineage(
        2,
        source_set.entity_set_id,
        target_set.entity_set_id,
        np.asarray((1, 0), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int64),
        np.full((2,), meshing.EntityLineageKind.COLLAPSED_INTO, dtype=np.int32),
    )
    collapsed_lineage = meshing.MeshLineage(
        source.topology_id, target.topology_id, (only_collapsed,)
    )
    with pytest.raises(ValueError):
        inherit_scope(dominant, collapsed_lineage, target, "material")
    all_members = dominant.union(collapsed)
    np.testing.assert_array_equal(
        inherit_scope(
            all_members, collapsed_lineage, target, "material"
        ).global_entity_ids,
        (0,),
    )


def test_sparse_semantic_inventory_queries_handle_an_empty_real_rank() -> None:
    environment = dict(os.environ)
    environment.update(
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        PYTHONPATH=str(Path(phx.__file__).resolve().parents[1]),
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _SPARSE_PACKET_SCENARIO, address, str(rank)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for rank in range(2)
    ]
    try:
        outputs = [process.communicate(timeout=120) for process in processes]
    finally:
        for process in processes:
            process.kill()
            process.wait(timeout=10)
    for process, (stdout, stderr) in zip(processes, outputs, strict=True):
        if process.returncode:
            pytest.fail(f"{stdout}\n{stderr}", pytrace=False)


_SPARSE_PACKET_SCENARIO = textwrap.dedent(
    """
    import sys
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    jax.distributed.initialize(coordinator_address=sys.argv[1],num_processes=2,
        process_id=int(sys.argv[2]),initialization_timeout=60)
    from phydrax.meshing._scope import _local_logical_lookup
    mesh = Mesh(np.asarray(jax.devices()),("inventory",))
    sharding = NamedSharding(mesh,PartitionSpec("inventory"))
    scalar_ids = np.asarray((10,20,30,40),dtype=np.int64)
    scalar_values = np.asarray((100,200,300,400),dtype=np.int32)
    identifiers = jax.make_array_from_callback((4,),sharding,lambda index:scalar_ids[index])
    values = jax.make_array_from_callback((4,),sharding,lambda index:scalar_values[index])
    queries = jnp.asarray((),dtype=jnp.int64) if jax.process_index()==0 else jnp.asarray((20,99,40),dtype=jnp.int64)
    selected, valid = _local_logical_lookup(identifiers,values,queries)
    membership = _local_logical_lookup(identifiers,None,queries)[1]
    key_ids = np.asarray(((10,1),(10,2),(20,1),(30,4)),dtype=np.int64)
    key_sharding = NamedSharding(mesh,PartitionSpec("inventory",None))
    keys = jax.make_array_from_callback((4,2),key_sharding,lambda index:key_ids[index])
    key_queries = jnp.empty((0,2),dtype=jnp.int64) if jax.process_index()==0 else jnp.asarray(((10,2),(99,0),(30,4)),dtype=jnp.int64)
    key_values, key_valid = _local_logical_lookup(keys,values,key_queries)
    if jax.process_index()==0:
        np.testing.assert_array_equal(selected,np.empty((0,),dtype=np.int32))
        np.testing.assert_array_equal(valid,np.empty((0,),dtype=np.bool_))
        np.testing.assert_array_equal(membership,np.empty((0,),dtype=np.bool_))
        np.testing.assert_array_equal(key_values,np.empty((0,),dtype=np.int32))
        np.testing.assert_array_equal(key_valid,np.empty((0,),dtype=np.bool_))
    else:
        np.testing.assert_array_equal(valid,(True,False,True))
        np.testing.assert_array_equal(membership,valid)
        np.testing.assert_array_equal(selected[valid],(200,400))
        np.testing.assert_array_equal(key_valid,(True,False,True))
        np.testing.assert_array_equal(key_values[key_valid],(200,400))
    jax.distributed.shutdown()
    """
)


_SCENARIO = textwrap.dedent(
    """
    import sys
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np
    import phydrax as phx
    from phydrax.discretization._adaptive_simplex import coarsen_adaptive_simplex_parts
    from phydrax.meshing._organization import validate_local_mesh_organization
    from phydrax.meshing.providers._native_publication import NativeCertificationRequest, publish_native_result

    M, D = phx.meshing, phx.discretization
    points = np.asarray(((0,0),(1,0),(0,1),(3,0),(4,0),(3,1)), dtype=np.float64)
    cells = np.asarray(((0,1,2),(3,4,5)), dtype=np.int32)
    seed = M.certify_cell_mesh(D.CellMesh.from_triangles(points,cells), phx.SpatialCoordinateContract.si())
    mesh = seed.mesh
    scope = M.MeshingScope(mesh.mesh_id, mesh.numeric_version, M.MeshingEntityKind.MESH,
                          2, mesh.entity_set(2).entity_set_id, np.asarray((1,), dtype=np.int64))
    label = M.MeshLabel("remote-cell", scope)
    zone = M.MeshZone("remote-material", M.MeshZoneRole.REGION, scope,
                      material_id="water", region_role=M.RegionRole.FLUID)
    boundary = np.asarray(((0,1),(1,2),(2,0),(3,4),(4,5),(5,3)), dtype=np.int32)
    domain = phx.geometry.PiecewiseLinearDomain(points,boundary,
        np.tile(np.asarray((0,-1), dtype=np.int32),(boundary.shape[0],1)),
        ("domain",), source_id="disjoint-triangles")
    source = publish_native_result(mesh,seed.coordinate_contract,
        M.MeshingComplianceReport("disjoint-triangles"),(),seed.provider,
        {"kind":"independent-domain-certified-source"},
        NativeCertificationRequest(M.MeshCertificationSchedule("volume_plc"),
            "disjoint-triangles","r0",M.MeshingLimits(),domain=domain,
            cell_regions=np.zeros((2,), dtype=np.int32)),
        labels=(label,),zones=(zone,),
        audit_policy=M.CellMeshAuditPolicy(watertight_boundary=M.CellMeshAuditDisposition.REJECT),
        derivative_mode=M.MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
        enforced_limits=(),unenforced_limits=())
    partition_policy = M.MeshPartitionPolicy(M.MeshPartitionKind.MORTON,2,halo_width=2)
    distribution = M.prepare_mesh_distribution(M.MeshPart("domain",source),policy=partition_policy)
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=64,cell_capacity=64),
        distribution=distribution,partition_policy=partition_policy,
        audit_policy=M.CellMeshAuditPolicy(watertight_boundary=M.CellMeshAuditDisposition.REJECT))
    prepared = M.prepare_adaptive_simplex(source,policy=policy)
    partitioned = M.partition_adaptive_simplex(prepared)
    marks = partitioned.cell_marks(np.asarray((0,1),dtype=np.int64))
    states = D.refine_adaptive_simplex_parts(partitioned.layout,partitioned.parts,partitioned.states,marks).state
    if sys.argv[1] == "coarsen":
        states = coarsen_adaptive_simplex_parts(partitioned.layout,partitioned.parts,states,states.mesh.cell_active).state
    results = M.commit_partitioned_adaptive_simplex(partitioned,states)
    expected = np.asarray((1,) if sys.argv[1] == "coarsen" else (4,5),dtype=np.int64)
    for result in results:
        target = result.target
        scope = target.labels[0].scope
        np.testing.assert_array_equal(scope.global_entity_ids,expected)
        np.testing.assert_array_equal(target.zones[0].scope.global_entity_ids,expected)
        assert target.zones[0].material_id == "water"
        assert target.zones[0].region_role is M.RegionRole.FLUID
        selected = M.resolve_mesh_scope(target.mesh,scope)
        np.testing.assert_array_equal(np.sort(np.asarray(target.mesh.entity_set(2).entity_ids)[np.asarray(selected.mask)]),scope.entity_ids)
        assert scope.local_coverage_id == target.mesh.storage.evidence_id
        assert target.labels[0].label_id == results[0].target.labels[0].label_id
        assert scope.union(scope).scope_id == scope.scope_id
        np.testing.assert_array_equal(scope.union(scope).entity_ids,scope.entity_ids)
    assert results[0].target.labels[0].scope.entity_ids.size == 0
    np.testing.assert_array_equal(results[1].target.labels[0].scope.entity_ids,expected)
    target = results[1].target
    arrays = dict(target.mesh.storage.logical_arrays)
    def rejected(patches: tuple[M.MeshPatch, ...], zones: tuple[M.MeshZone, ...],
                 labels: tuple[M.MeshLabel, ...], attributes: tuple[M.MeshAttribute, ...],
                 reason: str) -> None:
        try:
            validate_local_mesh_organization(source,target.mesh,arrays,patches,zones,labels,attributes)
        except ValueError:
            return
        raise AssertionError(reason)
    altered = M.MeshZone(target.zones[0].name,target.zones[0].role,target.zones[0].scope,
                         material_id="oil",region_role=M.RegionRole.FLUID)
    rejected(target.patches,(altered,),target.labels,target.attributes,"Changed scientific material was accepted")
    rejected(target.patches,target.zones,(),target.attributes,"A globally defined label was dropped")
    wrong_scope = eqx.tree_at(lambda value:value.entity_owner,target.labels[0].scope,
                             jnp.zeros(target.labels[0].scope.entity_owner.shape,dtype=jnp.int32))
    wrong_label = eqx.tree_at(lambda value:value.scope,target.labels[0],wrong_scope)
    rejected(target.patches,target.zones,(wrong_label,),target.attributes,"Disjoint owner placement was accepted")
    empty_scope = results[0].target.zones[0].scope
    try:
        M.validate_mesh_zones((
            M.MeshZone("first",M.MeshZoneRole.BOUNDARY,empty_scope),
            M.MeshZone("second",M.MeshZoneRole.BOUNDARY,empty_scope),
        ))
    except ValueError:
        pass
    else:
        raise AssertionError("Globally overlapping zones escaped through locally empty views")
    print("Exact remote material membership and negative inventory boundaries passed",sys.argv[1])
    """
)
