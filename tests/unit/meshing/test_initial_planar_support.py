# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Real native initial planar theorems remain authoritative PLC source premises."""

import inspect
import socket
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._association import PlcAssociationTransfer
from phydrax.meshing._association_routes import incidence_pairs
from phydrax.meshing._device_adaptation import prepare_partitioned_mesh_adaptation
from phydrax.meshing._distributed_planar_generation import (
    PlanarTileSchedule,
    prepare_distributed_planar_generation,
)
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._initial_certification import (
    InitialCollectiveMeshEvidence,
    InitialPlanarPremise,
)
from phydrax.meshing._plc_planar_support import (
    restart_planar_associations,
    restart_planar_lineage,
)
from phydrax.meshing._result import require_original_meshing_source
from phydrax.meshing.providers._native_planar_domain import source_arrays
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices
from tools.meshing_qualification.scenarios import _distributed_euler as euler_history


def _native_planar_initial(
    *, offset: float = 0.0, width: float = 2.0, revision: str = "authored",
) -> tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer]:
    meshing = phx.meshing
    # A translated nonsquare rectangle exercises the actual authored domain,
    # rather than a special coordinate frame or the qualification unit square.
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((2.0, 3.0), (2.0 + width, 3.0), (2.0 + width, 4.0), (2.0, 4.0)))
        + offset,
        ((0, 1, 2, 3),),
        feature_id="initial-planar-support",
    )
    # The revision names the authored source strata, including its region label.
    author = meshing.NativePlanarSource(region, revision)
    scope = meshing.MeshingScope(
        author.source_id,
        author.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        "initial-planar-support-region",
        np.asarray((0,), dtype=np.int64),
    )
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(
            2, 2, meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            meshing.UniformSizeControl(
                scope, 0.5, strength=meshing.SizeControlStrength.SOFT
            ),
        ),
        quality_target=meshing.MeshQualityTarget(minimum_angle=0.25, hard=True),
    )
    preparation = prepare_distributed_planar_generation(
        author,
        specification,
        PlanarTileSchedule(),
        meshing.DeviceGenerationLayout(
            vertex_capacity=1000,
            cell_capacity=2000,
            candidate_capacity=64,
            maximum_work_units=2_000_000,
        ),
    )
    source = (
        meshing.NativeMeshingProvider(
            meshing.NativeMeshingOptions("planar_constrained_delaunay")
        )
        .plan(
            author,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
            initial_partition=preparation,
        )
        .execute()
    )
    domain = preparation.tiles.region_domain()
    _, edges, _ = source_arrays(author)
    transfer = PlcAssociationTransfer(
        domain,
        source.coordinate_contract,
        author.source_revision,
        edge_vertices=edges,
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=domain.facet_regions,
    )
    return source, transfer


@pytest.fixture(scope="module")
def planar_initial() -> tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer]:
    return _native_planar_initial()


def test_initial_planar_source_proves_support_without_serial_certificates(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
) -> None:
    source, transfer = planar_initial
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    assert isinstance(evidence.premise, InitialPlanarPremise)
    assert require_original_meshing_source(source) is evidence
    assert source.certification is None
    assert source.collective_certificates is None
    assert any(
        leaf is evidence.premise.tile_owners
        for leaf in jax.tree_util.tree_leaves(evidence.premise)
    )
    # Association signs describe the published edge's incidence order, not the
    # receipt pair already oriented along the authored source segment.
    edge_set = source.mesh.entity_set(1)
    edge_association = next(
        value
        for value in source.associations
        if value.target_entity_set_id == edge_set.entity_set_id
    )
    connectivity = source.mesh.connectivity
    assert isinstance(connectivity, phx.discretization.PolygonalConnectivity)
    edge_ids = np.asarray(edge_set.entity_ids)
    order = np.argsort(edge_ids)
    rows = order[
        np.searchsorted(edge_ids[order], np.asarray(edge_association.target_global_ids))
    ]
    endpoints = np.asarray(source.mesh.coordinates)[
        np.asarray(connectivity.edges)[rows]
    ]
    authored = transfer.source_vertices[
        transfer.edge_vertices[np.asarray(edge_association.source_indices)]
    ]
    expected_orientations = np.sign(
        np.sum(
            (endpoints[:, 1] - endpoints[:, 0]) * (authored[:, 1] - authored[:, 0]),
            axis=1,
        )
    ).astype(np.int8)
    assert np.any(expected_orientations < 0)
    np.testing.assert_array_equal(
        edge_association.orientations, expected_orientations
    )
    with NativeExecutionScope(evidence.premise.specification.limits.native_limits()):
        vertex, dimensions = transfer.source_associations(source)
        levels = transfer.classes(source)
    assert vertex is not None
    assert dimensions == (1, 2)
    assert len(levels) == 3
    assert np.all(levels[2].dimensions == 2)
    assert np.all(levels[2].indices == 0)
    assert np.any(levels[1].dimensions == 1)
    assert np.any(levels[1].dimensions == 2)


@pytest.mark.parametrize("fault", (
    "none", "missing-owner", "duplicate-owner", "missing-associations",
    "stale-geometry", "stale-association", "missing-evidence",
    "foreign-owner", "missing-owned-row",
))
def test_restart_planar_archive_rows_are_complete_and_retryable(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
    fault: str,
) -> None:
    source, transfer = planar_initial
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    historical = (source,)
    if fault == "missing-owner":
        historical = ()
    elif fault == "duplicate-owner":
        historical = (source, source)
    elif fault == "foreign-owner":
        foreign, _ = _native_planar_initial(offset=0.25)
        historical = (foreign,)
    elif fault == "missing-owned-row":
        storage = source.mesh.storage
        assert storage is not None
        owned = list(storage.entity_owned)
        owned[0] = owned[0].at[0].set(False)
        changed = eqx.tree_at(lambda value: value.entity_owned, storage, tuple(owned))
        mesh = eqx.tree_at(lambda value: value.storage, source.mesh, changed)
        historical = (eqx.tree_at(lambda value: value.mesh, source, mesh),)
    elif fault == "missing-associations":
        historical = (eqx.tree_at(lambda value: value.associations, source, ()),)
    elif fault == "stale-geometry":
        geometry = eqx.tree_at(
            lambda value: value.coordinates, source.geometry,
            source.geometry.coordinates.at[0, 0].add(0.125),
        )
        historical = (eqx.tree_at(lambda value: value.geometry, source, geometry),)
    elif fault == "missing-evidence":
        historical = (eqx.tree_at(lambda value: value.collective_evidence, source, None),)
    elif fault == "stale-association":
        association = next(
            value for value in source.associations
            if value.target_entity_set_id == source.mesh.entity_set(1).entity_set_id
        )
        changed = eqx.tree_at(
            lambda value: value.source_indices, association,
            (association.source_indices + 1) % transfer.edge_indices.size,
        )
        associations = tuple(
            changed if value is association else value for value in source.associations
        )
        historical = (eqx.tree_at(lambda value: value.associations, source, associations),)
    limits = evidence.premise.specification.limits
    with NativeExecutionScope(limits.native_limits()):
        with coordinate_reference_budget(limits, queries_upper=transfer.maximum_support_queries) as budget:
            restart_planar_associations(
                transfer, (source,), source, source.mesh, evidence, budget=budget,
            )
            if fault != "none":
                with pytest.raises(MeshingFailure):
                    restart_planar_associations(
                        transfer, historical, source, source.mesh, evidence, budget=budget,
                    )
            restored = restart_planar_associations(
                transfer, (source,), source, source.mesh, evidence, budget=budget,
            )
    assert source.collective_evidence is evidence
    by_target = {value.target_entity_set_id: value for value in restored}
    for old in source.associations:
        new = by_target[old.target_entity_set_id]
        rows = new.target_rows(np.asarray(old.target_global_ids))
        for name in ("source_dimensions", "source_indices", "parameters", "orientations"):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name)[rows])


def test_restart_planar_archive_reconstruction_preserves_rearchive_ids(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
    tmp_path: Path,
) -> None:
    source, transfer = planar_initial
    receipt = write_meshing_source_closure(tmp_path / "source", (source, transfer))
    restored, restored_transfer = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id,
    )
    assert restored is not source
    assert restored.result_id == source.result_id
    evidence = restored.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    limits = evidence.premise.specification.limits
    with NativeExecutionScope(limits.native_limits()):
        with coordinate_reference_budget(limits, queries_upper=transfer.maximum_support_queries) as budget:
            associations = restart_planar_associations(
                restored_transfer, (restored,), restored, restored.mesh, evidence,
                budget=budget,
            )
    by_target = {value.target_entity_set_id: value for value in associations}
    for old in restored.associations:
        new = by_target[old.target_entity_set_id]
        rows = new.target_rows(np.asarray(old.target_global_ids))
        np.testing.assert_array_equal(old.source_indices, new.source_indices[rows])
        np.testing.assert_array_equal(old.parameters, new.parameters[rows])
    rearchived = write_meshing_source_closure(
        tmp_path / "rearchived", (restored, restored_transfer),
    )
    assert rearchived.content_id == receipt.content_id


@pytest.mark.parametrize("fault", (
    "none", "missing", "duplicate", "foreign-source-set", "foreign-target-set",
    "mutated-lineage", "mutated-source", "mutated-target", "not-preserved",
))
def test_restart_planar_lineage_reconstructs_and_refuses_mutation(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
    fault: str,
) -> None:
    from phydrax.meshing._lineage_records import (
        EntityLineage,
        EntityLineageKind,
        MeshLineage,
    )

    historical, _ = planar_initial
    mesh = historical.mesh

    def lineage(*, invalid: bool) -> MeshLineage:
        records = []
        for dimension in range(3):
            entity_set = mesh.entity_set(dimension)
            # The producer's row order is deliberately unrelated to mesh order.
            ids = np.asarray(entity_set.entity_ids, dtype=np.int64)[::-1]
            source_set = target_set = entity_set.entity_set_id
            kinds = np.full(ids.shape, EntityLineageKind.PRESERVED, dtype=np.int32)
            if invalid and dimension == 0:
                if fault == "missing":
                    ids, kinds = ids[:-1], kinds[:-1]
                elif fault == "duplicate":
                    ids, kinds = np.concatenate((ids, ids[:1])), np.append(kinds, 0)
                elif fault == "foreign-source-set":
                    source_set = "foreign-source"
                elif fault == "foreign-target-set":
                    target_set = "foreign-target"
                elif fault == "not-preserved":
                    kinds[0] = EntityLineageKind.REFINED_FROM
            record = EntityLineage(dimension, source_set, target_set, ids, ids, kinds)
            if invalid and dimension == 0:
                if fault in ("mutated-lineage", "mutated-source"):
                    record = eqx.tree_at(
                        lambda value: value.source_global_ids, record,
                        record.source_global_ids.at[0].add(100000),
                    )
                elif fault == "mutated-target":
                    record = eqx.tree_at(
                        lambda value: value.target_global_ids, record,
                        record.target_global_ids.at[0].add(100000),
                    )
            records.append(record)
        return MeshLineage(mesh.topology_id, mesh.topology_id, tuple(records))

    accepted = lineage(invalid=False)
    if fault != "none":
        with pytest.raises(MeshingFailure):
            restart_planar_lineage(historical, lineage(invalid=True), mesh, mesh)
    current = restart_planar_lineage(historical, accepted, mesh, mesh)
    assert current is not accepted
    for dimension in range(3):
        old, new = accepted.entity_lineage(dimension), current.entity_lineage(dimension)
        assert new is not old
        np.testing.assert_array_equal(new.source_global_ids, old.source_global_ids)
        np.testing.assert_array_equal(new.target_global_ids, old.target_global_ids)
        assert new.source_entity_set_id == mesh.entity_set(dimension).entity_set_id
        assert new.target_entity_set_id == mesh.entity_set(dimension).entity_set_id


def test_cold_scope_replay_retains_each_actual_result_graph(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
    tmp_path: Path,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_sources import _meshing_source_replay_plan
    from phydrax.meshing._scope import MeshScopeProjection

    source, _ = planar_initial
    archive = write_meshing_source_closure(tmp_path / "scope-owners", source)
    owners = tuple(
        read_meshing_source_closure(archive.path, expected_content_id=archive.content_id)
        for _ in range(2)
    )
    assert owners[0] is not owners[1]
    assert owners[0].result_id == owners[1].result_id == source.result_id
    plan = _meshing_source_replay_plan(
        tuple({"collective_target": owner} for owner in owners),
        {}, DEFAULT_ARRAY_ARCHIVE_LIMITS, None,
    )
    for owner in owners:
        evidence = owner.collective_evidence
        for record in (*owner.patches, *owner.zones, *owner.labels):
            projection = record.scope.projection
            assert projection is not None
            assert plan.scope_evidence[id(projection)] is evidence
            MeshScopeProjection(
                evidence, projection.publication, projection.source_revision,
                projection.dimension, projection.membership_name,
                exclusive_scopes=projection.exclusive_scopes,
            )
            foreign = owners[1] if owner is owners[0] else owners[0]
            with pytest.raises(ValueError, match="different bank objects"):
                type(projection.publication).require_source(
                    projection.publication, foreign.collective_evidence.logical_arrays,
                )


def test_cold_restart_publication_uses_current_certified_plc_predecessor(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._execution_runtime import ExecutionRuntime
    from phydrax.meshing import _plc_planar_support as support
    from phydrax.meshing._device_adaptation import (
        commit_partitioned_mesh_restart,
        restore_partitioned_mesh_state,
    )
    from phydrax.meshing._restart_distribution import (
        repack_simplex_restart,
        SimplexRestartPlacement,
        SimplexRestartRepackProof,
    )
    from phydrax.meshing._result import CollectiveMeshEvidence

    source, transfer = planar_initial
    partition = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.PROVIDER, 1, halo_width=1,
    )
    distribution = phx.meshing.prepare_mesh_distribution(
        phx.meshing.MeshPart("domain", source), policy=partition,
    )
    accepted, distribution, _ = _planar_owner_epoch(
        source, transfer, distribution, partition,
        phx.discretization.refine_adaptive_simplex_parts,
    )
    archive = write_meshing_source_closure(
        tmp_path / "accepted", (accepted, transfer, distribution),
    )
    historical, transfer, distribution = read_meshing_source_closure(
        archive.path, expected_content_id=archive.content_id,
    )
    assert historical.result_id == accepted.result_id
    storage, evidence = historical.mesh.storage, historical.collective_evidence
    assert storage is not None and isinstance(evidence, CollectiveMeshEvidence)
    layout, saved, _ = restore_partitioned_mesh_state(storage, evidence)
    cells = jnp.where(saved.mesh.cell_ids >= 0, 0, -1).astype(jnp.int32)
    vertices = jnp.where(saved.mesh.vertex_ids >= 0, 0, -1).astype(jnp.int32)
    receipt = repack_simplex_restart(
        ExecutionRuntime.current().root_group, layout, saved, cells, vertices,
        jnp.zeros_like(cells), source_result=historical,
    )
    proof = SimplexRestartRepackProof(receipt)
    np.testing.assert_array_equal(receipt.invalid_reason_masks, [0])
    saved_receipt = replace(
        receipt,
        parts=SimplexRestartPlacement(
            receipt.parts.axis_name, receipt.parts.part_count, receipt.parts.neighbor_pairs,
        ),
    )
    receipt_archive = write_meshing_source_closure(
        tmp_path / "restart-receipt", (historical, transfer, distribution, saved_receipt),
    )
    _, _, _, restored_receipt = read_meshing_source_closure(
        receipt_archive.path, expected_content_id=receipt_archive.content_id,
    )
    np.testing.assert_array_equal(
        restored_receipt.invalid_reason_masks, receipt.invalid_reason_masks,
    )
    np.testing.assert_array_equal(
        restored_receipt.duplicate_record_conflicts, receipt.duplicate_record_conflicts,
    )
    restored_proof = SimplexRestartRepackProof(restored_receipt, collective=True)
    assert restored_proof.content_id == proof.content_id
    tampered = replace(
        receipt, invalid_reason_masks=jnp.ones_like(receipt.invalid_reason_masks),
    )
    with pytest.raises(ValueError, match="different array bindings"):
        proof.require_receipt(tampered)
    with pytest.raises(ValueError, match="different numerical content"):
        SimplexRestartRepackProof(tampered)
    tampered_conflict = replace(
        receipt, duplicate_record_conflicts=receipt.duplicate_record_conflicts.at[0, 0].set(1),
    )
    with pytest.raises(ValueError, match="different array bindings"):
        proof.require_receipt(tampered_conflict)
    with pytest.raises(ValueError, match="different numerical content"):
        SimplexRestartRepackProof(tampered_conflict)
    prepared = phx.meshing.prepare_mesh_adaptation(
        historical, phx.meshing.MarkedMeshAdaptation(),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=phx.discretization.AdaptiveSimplexPolicy(
                vertex_capacity=1000, cell_capacity=2000,
            ),
            distribution=distribution, partition_policy=partition,
            association_transfer=transfer,
        ),
    )
    query_sources = []
    propagate = support.planar_propagate

    def observe(*args: Any, **kwargs: Any) -> Any:
        query_sources.append(args[1])
        for dimension in range(3):
            record = args[2].entity_lineage(dimension)
            assert record.source_entity_set_id == args[1].mesh.entity_set(dimension).entity_set_id
            assert record.target_entity_set_id == args[3].entity_set(dimension).entity_set_id
        return propagate(*args, **kwargs)

    monkeypatch.setattr(support, "planar_propagate", observe)
    (result,) = commit_partitioned_mesh_restart(
        receipt, prepared, proof=proof, historical_sources=(historical,),
    ).local_results
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert len(query_sources) == 1
    current = query_sources[0]
    assert current is not historical
    assert current.mesh is result.target.mesh
    assert current.geometry is result.target.geometry
    assert current.collective_evidence is result.target.collective_evidence
    assert current.collective_certificates is not None
    assert receipt.source_result is historical
    assert historical.result_id == accepted.result_id
    assert result.lineage is not None
    for fault in ("missing", "duplicate", "foreign"):
        archived = result.lineage
        if jax.process_index() == 0:
            record = archived.entity_lineage(0)
            ids = record.source_global_ids
            changed_ids = (
                ids[:-1] if fault == "missing"
                else jnp.concatenate((ids, ids[:1])) if fault == "duplicate"
                else ids.at[0].add(100000)
            )
            changed = eqx.tree_at(
                lambda value: value.source_global_ids, record, changed_ids,
            )
            archived = eqx.tree_at(
                lambda value: value.entities, archived,
                (changed, *archived.entities[1:]),
            )
        with pytest.raises(MeshingFailure):
            restart_planar_lineage(historical, archived, current.mesh, result.target.mesh)
        renewed = restart_planar_lineage(
            historical, result.lineage, current.mesh, result.target.mesh,
        )
        assert renewed.source_topology_id == current.mesh.topology_id


def test_initial_planar_associations_dispatch_without_binding_source_dags(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, transfer = planar_initial
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    getattribute = eqx.Module.__getattribute__
    methods = {
        "require_passed", "require", "validate_target", "entity_set", "roles",
        "target_rows",
    }

    def reject_bound_source_method(value: Any, name: str) -> Any:
        if name in methods:
            frame = inspect.currentframe()
            assert frame is not None and frame.f_back is not None
            caller = frame.f_back.f_code
            del frame
            assert not (
                caller.co_name == "planar_source_associations"
                and caller.co_filename.endswith("/_plc_planar_support.py")
            ), f"Source authentication must not bind its retained DAG for {name}"
        return getattribute(value, name)

    monkeypatch.setattr(eqx.Module, "__getattribute__", reject_bound_source_method)
    with NativeExecutionScope(evidence.premise.specification.limits.native_limits()):
        vertex, dimensions = type(transfer).source_associations(transfer, source)
    assert any(vertex is association for association in source.associations)
    assert dimensions == (1, 2)


def test_initial_planar_source_refines_and_coarsens_with_target_certificates(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
) -> None:
    source, transfer = planar_initial
    storage = source.mesh.storage
    assert storage is not None
    meshing = phx.meshing
    partition = meshing.MeshPartitionPolicy(
        meshing.MeshPartitionKind.PROVIDER,
        storage.partition_count,
        halo_width=storage.neighborhood_depth,
    )
    distribution = meshing.prepare_mesh_distribution(
        meshing.MeshPart("domain", source), policy=partition
    )
    original = require_original_meshing_source(source)
    assert isinstance(original, InitialCollectiveMeshEvidence)
    counts = [storage.global_entity_counts[2]]
    system = phx.equations.EulerSystem(2)
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        source.mesh,
        component_names=system.component_names,
        neighborhood_complete=storage.local_neighborhood_complete,
        physical_boundary_faces=storage.local_physical_boundary_facets,
        closure_evidence_id=storage.evidence_id,
    ).prepare()
    np.testing.assert_array_equal(
        geometry.cell_global_ids, source.mesh.entity_set(2).entity_ids
    )
    centers = geometry.cell_centers
    original_values = system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + 0.1 * centers[:, 0],
                0.2 * centers[:, 0],
                -0.1 * centers[:, 1],
                1.0 + 0.1 * centers[:, 1],
            ),
            axis=1,
        )
    )
    original_inventory = np.sum(
        np.asarray(geometry.cell_volumes)[:, None] * np.asarray(original_values), axis=0
    )
    euler_values = original_values
    current = source
    for operation in (
        phx.discretization.refine_adaptive_simplex_parts,
        phx.discretization.coarsen_adaptive_simplex_parts,
    ):
        policy = meshing.MeshAdaptationPolicy(
            meshing.MeshAdaptationRoute.DEVICE_BISECTION,
            compatibility=meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
            device_policy=phx.discretization.AdaptiveSimplexPolicy(
                vertex_capacity=1000, cell_capacity=2000
            ),
            distribution=distribution,
            partition_policy=partition,
            association_transfer=transfer,
        )
        prepared = prepare_partitioned_mesh_adaptation(
            meshing.prepare_mesh_adaptation(
                current, meshing.MarkedMeshAdaptation(), policy=policy
            )
        )
        assert prepared.prepared.adaptation.policy.association_transfer is transfer
        updated = operation(
            prepared.layout,
            type(prepared).execution_parts(prepared),
            prepared.states,
            prepared.states.mesh.cell_active,
        )
        (accepted,) = meshing.commit_partitioned_adaptive_simplex(
            prepared, updated.state
        ).local_results
        assert accepted.status is meshing.MeshAdaptationStatus.COMPLETE
        moved, transfer_record = euler_history.transfer_history(
            accepted,
            prepared.states,
            updated.state,
            euler_values,
            tuple(prepared.parts.mesh.devices.flat),
            message_capacity=2000,
            system=system,
        )
        defect = transfer_record["maximum_inventory_defect"]
        assert isinstance(defect, float) and defect <= 1.0e-12
        if operation is phx.discretization.refine_adaptive_simplex_parts:
            np.testing.assert_array_equal(moved, euler_values[accepted.parent_cells])
        euler_values = moved
        current = accepted.target
        root = require_original_meshing_source(current)
        assert isinstance(root, InitialCollectiveMeshEvidence)
        assert root.evidence_id == original.evidence_id
        assert current.collective_certificates is not None
        embedding, coverage = current.collective_certificates
        embedding.binding.require(current.mesh, current.geometry)
        coverage.binding.require(current.mesh, current.geometry)
        assert embedding.status == coverage.status == "certified"
        assert coverage.domain_id == transfer.domain.domain_id
        assert original.source_evidence_id in coverage.premise_certificate_ids
        assert current.mesh.storage is not None
        assert current.mesh.storage.evidence_id in coverage.premise_certificate_ids
        counts.append(current.mesh.storage.global_entity_counts[2])
        with NativeExecutionScope(original.premise.specification.limits.native_limits()):
            transfer.classes(current)
        assert accepted.distribution is not None
        distribution = accepted.distribution.target
        np.testing.assert_allclose(
            euler_history.inventory(current, euler_values, system=system),
            original_inventory,
            rtol=0.0,
            atol=1.0e-12,
        )
    assert counts[1] > counts[0]
    assert counts[2] < counts[1]
    np.testing.assert_allclose(euler_values, original_values, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("changed_bank", ("coordinates", "tile_weights", "tile_owners"))
def test_initial_planar_support_rejects_tampered_theorem_content(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
    changed_bank: str,
) -> None:
    source, transfer = planar_initial
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    assert isinstance(evidence.premise, InitialPlanarPremise)
    if changed_bank == "coordinates":
        banks = dict(evidence.logical_arrays)
        banks["coordinates"] = banks["coordinates"].at[0, 0].add(0.125)
        changed_evidence = eqx.tree_at(
            lambda value: value.logical_arrays,
            evidence,
            tuple(sorted(banks.items())),
        )
        failure, message = phx.meshing.MeshingFailure, "initial.content_id"
    elif changed_bank == "tile_weights":
        tiles = evidence.premise.tiles
        changed_tiles = eqx.tree_at(
            lambda value: value.tile_weights, tiles, tiles.tile_weights + 0.125
        )
        changed_premise = eqx.tree_at(
            lambda value: value.tiles, evidence.premise, changed_tiles
        )
        changed_evidence = eqx.tree_at(
            lambda value: value.premise, evidence, changed_premise
        )
        failure, message = ValueError, "changed retained banks"
    else:
        premise = evidence.premise
        changed_premise = eqx.tree_at(
            lambda value: value.tile_owners, premise, premise.tile_owners + 1
        )
        changed_evidence = eqx.tree_at(
            lambda value: value.premise, evidence, changed_premise
        )
        failure, message = ValueError, "premise_id"
    changed = eqx.tree_at(
        lambda value: value.collective_evidence, source, changed_evidence
    )
    with NativeExecutionScope(evidence.premise.specification.limits.native_limits()):
        with pytest.raises(failure, match=message):
            transfer.source_associations(changed)


def test_initial_planar_support_rejects_foreign_domain(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
) -> None:
    source, transfer = planar_initial
    domain = transfer.domain
    foreign_domain = phx.geometry.PiecewiseLinearDomain(
        np.asarray(domain.vertices) + np.asarray((0.25, 0.0)),
        domain.facets,
        domain.facet_regions,
        domain.region_ids,
        source_id=domain.source_id,
    )
    foreign = PlcAssociationTransfer(
        foreign_domain,
        transfer.coordinate_contract,
        transfer.source_revision,
        edge_vertices=transfer.edge_vertices,
        triangle_vertices=transfer.triangle_vertices,
        triangle_facets=transfer.triangle_facets,
        facet_regions=foreign_domain.facet_regions,
    )
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    with NativeExecutionScope(evidence.premise.specification.limits.native_limits()):
        with pytest.raises(ValueError, match="represented source, revision and domain"):
            foreign.source_associations(source)


def test_initial_planar_support_rejects_changed_coordinate_maps(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
) -> None:
    source, transfer = planar_initial
    changed_geometry = eqx.tree_at(
        lambda value: value.coordinates,
        source.geometry,
        source.geometry.coordinates.at[0, 0].add(0.125),
    )
    changed = eqx.tree_at(lambda value: value.geometry, source, changed_geometry)
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    with NativeExecutionScope(evidence.premise.specification.limits.native_limits()):
        with pytest.raises(ValueError, match="coordinate|coefficient|storage|carrier"):
            transfer.source_associations(changed)


def planar_initial_owner_support() -> dict[str, int | str]:
    """Entry point for the real two-process closure regression."""
    source, transfer = _native_planar_initial()
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    with NativeExecutionScope(evidence.premise.specification.limits.native_limits()):
        levels = transfer.classes(source)
    links = incidence_pairs(source.mesh, 1, 2)
    local_incidence = np.bincount(
        links[:, 0], minlength=source.mesh.entity_set(1).count
    )
    return {
        "owner": jax.process_index(),
        "source": source.result_id,
        "truncated_region_edges": int(
            np.count_nonzero((levels[1].dimensions == 2) & (local_incidence == 1))
        ),
    }


def test_initial_planar_support_proves_two_owner_truncated_closures() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(process_id: int) -> dict[str, Any]:
        return run_on_forced_devices(
            "tests.unit.meshing.test_initial_planar_support",
            "planar_initial_owner_support",
            (),
            devices=1,
            env=gloo_rank_environment(coordinator, process_id),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(launch, range(2)))
    assert [owner["owner"] for owner in owners] == [0, 1]
    assert owners[0]["source"] == owners[1]["source"]
    assert all(owner["truncated_region_edges"] > 0 for owner in owners)


def _planar_owner_start(
    *, offset: float = 0.0, revision: str = "authored"
) -> tuple[
    phx.meshing.CellMeshingResult,
    PlcAssociationTransfer,
    phx.meshing.MeshDistribution,
    phx.meshing.MeshPartitionPolicy,
]:
    # Three unit tiles over two processes give one owner twice the closure.
    source, transfer = _native_planar_initial(
        width=3.0, offset=offset, revision=revision
    )
    storage = source.mesh.storage
    assert storage is not None
    meshing = phx.meshing
    # Provider ownership is kept as authored: its 2:1 tile split is declared,
    # not an automatic route that may rebalance.
    partition = meshing.MeshPartitionPolicy(
        meshing.MeshPartitionKind.PROVIDER,
        storage.partition_count,
        halo_width=storage.neighborhood_depth,
        maximum_imbalance=2.0,
    )
    distribution = meshing.prepare_mesh_distribution(
        meshing.MeshPart("domain", source), policy=partition
    )
    return source, transfer, distribution, partition


def _planar_owner_epoch(
    current: phx.meshing.CellMeshingResult,
    transfer: PlcAssociationTransfer,
    distribution: phx.meshing.MeshDistribution,
    partition: phx.meshing.MeshPartitionPolicy,
    operation: Callable[..., Any],
    *,
    limits: phx.meshing.MeshingLimits | None = None,
) -> tuple[phx.meshing.CellMeshingResult, phx.meshing.MeshDistribution, dict[str, Any]]:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.geometry import _collective_domain_coverage as coverage_owner

    meshing = phx.meshing
    prepared = prepare_partitioned_mesh_adaptation(
        meshing.prepare_mesh_adaptation(
            current,
            meshing.MarkedMeshAdaptation(),
            policy=meshing.MeshAdaptationPolicy(
                meshing.MeshAdaptationRoute.DEVICE_BISECTION,
                compatibility=meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
                limits=limits,
                device_policy=phx.discretization.AdaptiveSimplexPolicy(
                    vertex_capacity=1000, cell_capacity=2000
                ),
                distribution=distribution,
                partition_policy=partition,
                association_transfer=transfer,
            ),
        )
    )
    updated = operation(
        prepared.layout,
        prepared.parts,
        prepared.states,
        prepared.states.mesh.cell_active,
    )
    native_validity = coverage_owner._collective_validity
    certify_validity = coverage_owner.certify_cell_geometry_validity
    budgets: list[CoordinateEnclosureBudget] = []
    validity_calls = 0

    def observe_native_validity(*args: Any, **kwargs: Any) -> Any:
        ledger = kwargs.get("budget")
        assert isinstance(ledger, CoordinateEnclosureBudget)
        budgets.append(ledger)
        try:
            return native_validity(*args, **kwargs)
        finally:
            budgets.pop()

    def observe_certificate(*args: Any, **kwargs: Any) -> Any:
        nonlocal validity_calls
        assert budgets and kwargs.get("budget") is budgets[-1]
        validity_calls += 1
        # Observe the actual request; never replace its scientific verdict.
        return certify_validity(*args, **kwargs)

    with pytest.MonkeyPatch.context() as observer:
        observer.setattr(coverage_owner, "_collective_validity", observe_native_validity)
        observer.setattr(
            coverage_owner, "certify_cell_geometry_validity", observe_certificate
        )
        (accepted,) = meshing.commit_partitioned_adaptive_simplex(
            prepared, updated.state
        ).local_results
    assert validity_calls > 0
    target = accepted.target
    storage = target.mesh.storage
    assert storage is not None and target.collective_evidence is not None
    assert accepted.distribution is not None
    return (
        target,
        accepted.distribution.target,
        {
            "complete": accepted.status is meshing.MeshAdaptationStatus.COMPLETE,
            "result": target.result_id,
            "evidence": target.collective_evidence.evidence_id,
            "mesh": target.mesh.mesh_id,
            "stored": target.stored_content_id,
            "child_stored": getattr(
                target.collective_evidence, "stored_content_id", None
            ),
            "cells": storage.global_entity_counts[2],
            "resident_vertices": int(target.mesh.entity_set(0).count),
            "certified": target.collective_certificates is not None
            and all(
                certificate.status == "certified"
                for certificate in target.collective_certificates
            ),
        },
    )


def planar_initial_owner_epochs() -> dict[str, Any]:
    """Entry point: owner-local refine then coarsen over unequal resident closures.

    Variants change only the coarsening limits, the authored coordinates, or
    the authored region label of the same three-tile source."""
    refine = phx.discretization.refine_adaptive_simplex_parts
    coarsen = phx.discretization.coarsen_adaptive_simplex_parts
    source, transfer, distribution, partition = _planar_owner_start()
    refined, refined_distribution, first = _planar_owner_epoch(
        source, transfer, distribution, partition, refine
    )
    _, _, second = _planar_owner_epoch(
        refined, transfer, refined_distribution, partition, coarsen
    )
    _, _, limited = _planar_owner_epoch(
        refined,
        transfer,
        refined_distribution,
        partition,
        coarsen,
        limits=phx.meshing.MeshingLimits(maximum_cells=1_000_000),
    )
    variants = {"limits": limited}
    for name, offset, revision in (
        ("geometry", 0.25, "authored"),
        ("material", 0.0, "relabeled"),
    ):
        changed, changed_transfer, changed_distribution, changed_partition = (
            _planar_owner_start(offset=offset, revision=revision)
        )
        _, _, variants[name] = _planar_owner_epoch(
            changed, changed_transfer, changed_distribution, changed_partition, refine
        )
    return {"owner": jax.process_index(), "epochs": [first, second], "variants": variants}


def test_initial_planar_owner_epochs_share_collective_programs() -> None:
    """Unequal per-process closures still enter identically shaped collectives
    and publish one rank-invariant result identity bound to the current target."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(process_id: int) -> dict[str, Any]:
        return run_on_forced_devices(
            "tests.unit.meshing.test_initial_planar_support",
            "planar_initial_owner_epochs",
            (),
            devices=1,
            env=gloo_rank_environment(coordinator, process_id),
            timeout_seconds=900,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(launch, range(2)))
    assert [owner["owner"] for owner in owners] == [0, 1]
    refined, coarsened = zip(*(owner["epochs"] for owner in owners), strict=True)
    for epoch in (refined, coarsened):
        assert all(owner["complete"] and owner["certified"] for owner in epoch)
        for key in ("result", "evidence", "mesh", "cells"):
            assert epoch[0][key] == epoch[1][key], key
        # Each owner's storage anchor binds its resident certificate receipts.
        assert all(
            owner["stored"] and owner["stored"] != owner["child_stored"]
            for owner in epoch
        )
    assert coarsened[0]["cells"] < refined[0]["cells"]
    assert coarsened[0]["resident_vertices"] != coarsened[1]["resident_vertices"]
    for name, baseline in (
        ("limits", coarsened[0]),
        ("geometry", refined[0]),
        ("material", refined[0]),
    ):
        changed = [owner["variants"][name] for owner in owners]
        assert all(value["complete"] and value["certified"] for value in changed)
        assert changed[0]["result"] == changed[1]["result"], name
        assert changed[0]["evidence"] == changed[1]["evidence"], name
        assert changed[0]["cells"] == baseline["cells"], name
        assert changed[0]["result"] != baseline["result"], name
        assert changed[0]["evidence"] != baseline["evidence"], name
    # Limits leave the target carrier unchanged; only the bound policy differs.
    assert owners[0]["variants"]["limits"]["mesh"] == coarsened[0]["mesh"]
    assert owners[0]["variants"]["geometry"]["mesh"] != refined[0]["mesh"]


def test_initial_planar_support_rejects_foreign_initial_evidence(
    planar_initial: tuple[phx.meshing.CellMeshingResult, PlcAssociationTransfer],
) -> None:
    source, transfer = planar_initial
    foreign, _ = _native_planar_initial(offset=0.25)
    changed = eqx.tree_at(
        lambda value: value.collective_evidence, source, foreign.collective_evidence
    )
    evidence = source.collective_evidence
    assert isinstance(evidence, InitialCollectiveMeshEvidence)
    with NativeExecutionScope(evidence.premise.specification.limits.native_limits()):
        with pytest.raises(ValueError, match="published target"):
            transfer.source_associations(changed)


_OWNER_COMMIT_FAULTS = (
    "neighborhood",
    "source_cells",
    "source_vertices",
    "source_entities",
    "resource",
    "lineage",
    "unexpected_lineage",
    "premise_domain",
    "validity",
    "stencil",
    "transition",
)


def planar_initial_owner_faults(failed_owner: str) -> dict[str, Any]:
    """Real uneven closures must agree each refusal before the next collective."""
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.geometry import _collective_domain_coverage as coverage_owner
    from phydrax.meshing import (
        _collective_failure as failure_owner,
        _device_adaptation as device_owner,
    )

    owner = jax.process_index()
    failed_rank = int(failed_owner)
    source, transfer, distribution, partition = _planar_owner_start()
    refined, refined_distribution, first = _planar_owner_epoch(
        source,
        transfer,
        distribution,
        partition,
        phx.discretization.refine_adaptive_simplex_parts,
    )
    _, _, baseline = _planar_owner_epoch(
        refined,
        transfer,
        refined_distribution,
        partition,
        phx.discretization.coarsen_adaptive_simplex_parts,
    )
    faults = {}
    for mode in _OWNER_COMMIT_FAULTS:
        state = {"injected": False, "refusals": 0, "after_refusal": 0}
        trace = []
        source_lookups = 0

        def inject() -> None:
            if owner == failed_rank and not state["injected"]:
                state["injected"] = True
                if mode == "resource":
                    raise CoordinateEnclosureResourceError("coefficient_work", 0, 1, 0)
                if mode == "unexpected_lineage":
                    raise RuntimeError("unexpected owner-local lineage fault")
                raise ValueError(f"owner-local fault: {mode}")

        with pytest.MonkeyPatch.context() as patch:
            gather = multihost_utils.process_allgather
            decision = failure_owner.collective_failure_decision
            lookup = device_owner._dependency_phydrax_meshing__scope.local_logical_lookup

            def traced_gather(*args: Any, **kwargs: Any) -> Any:
                if state["refusals"]:
                    state["after_refusal"] += 1
                trace.append("gather")
                return gather(*args, **kwargs)

            def traced_decision(category: Any) -> Any:
                verdict = decision(category)
                if verdict.category is not None:
                    state["refusals"] += 1
                return verdict

            def faulted_lookup(*args: Any, **kwargs: Any) -> Any:
                nonlocal source_lookups
                frame = inspect.currentframe()
                assert frame is not None and frame.f_back is not None
                caller = frame.f_back.f_code.co_name
                del frame
                trace.append(f"lookup:{caller}")
                values, present = lookup(*args, **kwargs)
                if caller == "_source_closure":
                    source_lookups += 1
                selected = (
                    (mode == "source_cells" and caller == "_source_closure" and source_lookups == 1)
                    or (mode == "source_vertices" and caller == "_source_closure" and source_lookups == 2)
                    or (mode == "source_entities" and caller == "_source_entity_values")
                )
                if selected and owner == failed_rank and not state["injected"]:
                    assert present.size, "The regression must exercise a real source lookup"
                    state["injected"] = True
                    present = np.array(present, copy=True)
                    present.flat[0] = False
                return values, present

            def arm_after_coarsening(*args: Any, **kwargs: Any) -> Any:
                # Preparation and device execution remain genuine and unfaulted.
                # Install faults only in the publication that follows them.
                updated = phx.discretization.coarsen_adaptive_simplex_parts(*args, **kwargs)
                patch.setattr(multihost_utils, "process_allgather", traced_gather)
                patch.setattr(failure_owner, "collective_failure_decision", traced_decision)
                patch.setattr(
                    device_owner._dependency_phydrax_meshing__scope,
                    "local_logical_lookup",
                    faulted_lookup,
                )
                if mode == "neighborhood":
                    addressable = device_owner._host_part_array

                    def faulted_addressable(*args: Any, **kwargs: Any) -> Any:
                        frame = inspect.currentframe()
                        assert frame is not None and frame.f_back is not None
                        caller = frame.f_back
                        selected = (
                            caller.f_code.co_name == "host"
                            and caller.f_back is not None
                            and caller.f_back.f_code.co_name == "_neighborhood_commit"
                        )
                        del frame, caller
                        if selected:
                            inject()
                        return addressable(*args, **kwargs)

                    patch.setattr(device_owner, "_host_part_array", faulted_addressable)
                elif mode == "resource":
                    count = device_owner.bisection_host.uniform_count

                    def faulted_count(*args: Any, **kwargs: Any) -> Any:
                        # Charge the first real host publication operation. This
                        # fixture uses midpoint-DAG reverse support rather than
                        # the optional uniform-refinement ancestry branch.
                        inject()
                        return count(*args, **kwargs)

                    patch.setattr(device_owner.bisection_host, "uniform_count", faulted_count)
                elif mode in ("lineage", "unexpected_lineage"):
                    topology = device_owner._dependency_phydrax_meshing__topology_edit
                    entity_lineage = topology.entity_lineage

                    def faulted_lineage(*args: Any, **kwargs: Any) -> Any:
                        inject()
                        return entity_lineage(*args, **kwargs)

                    patch.setattr(topology, "entity_lineage", faulted_lineage)
                elif mode == "premise_domain":
                    current_domain = InitialPlanarPremise.require_current_domain

                    def faulted_domain(*args: Any, **kwargs: Any) -> Any:
                        inject()
                        return current_domain(*args, **kwargs)

                    patch.setattr(InitialPlanarPremise, "require_current_domain", faulted_domain)
                elif mode == "validity":
                    certify = coverage_owner.certify_cell_geometry_validity

                    def faulted_validity(*args: Any, **kwargs: Any) -> Any:
                        inject()
                        return certify(*args, **kwargs)

                    patch.setattr(coverage_owner, "certify_cell_geometry_validity", faulted_validity)
                elif mode in ("stencil", "transition"):
                    record = (
                        device_owner._dependency_phydrax_meshing__lineage_records.VertexInterpolationStencil
                        if mode == "stencil"
                        else device_owner._dependency_phydrax_meshing__lineage.CellMeshTransition
                    )
                    construct = record.__init__

                    def faulted_construct(*args: Any, **kwargs: Any) -> None:
                        inject()
                        construct(*args, **kwargs)

                    # Keep the real class and isinstance checks on healthy owners.
                    patch.setattr(record, "__init__", faulted_construct)
                return updated

            with pytest.raises(phx.meshing.MeshingFailure) as refused:
                _planar_owner_epoch(
                    refined,
                    transfer,
                    refined_distribution,
                    partition,
                    arm_after_coarsening,
                )
                pytest.fail(
                    f"{mode}: owner {owner} accepted the faulted epoch; "
                    f"injected={state['injected']}, refusal_count={state['refusals']}"
                )
            faults[mode] = {
                "category": refused.value.category.value,
                "injected": state["injected"],
                "cause": type(refused.value.__cause__).__name__,
                "refusals": state["refusals"],
                "after_refusal": state["after_refusal"],
                "trace": trace,
            }
    # Refusal must not poison the accepted source or the next genuine request.
    _, _, recovered = _planar_owner_epoch(
        refined,
        transfer,
        refined_distribution,
        partition,
        phx.discretization.coarsen_adaptive_simplex_parts,
    )
    return {
        "owner": owner,
        "refined": first,
        "baseline": baseline,
        "recovered": recovered,
        "faults": faults,
    }


@pytest.mark.parametrize("failed_owner", (0, 1))
def test_initial_planar_owner_commit_faults_are_agreed(failed_owner: int) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(process_id: int) -> dict[str, Any]:
        return run_on_forced_devices(
            "tests.unit.meshing.test_initial_planar_support",
            "planar_initial_owner_faults",
            (str(failed_owner),),
            devices=1,
            env=gloo_rank_environment(coordinator, process_id),
            timeout_seconds=900,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(launch, range(2)))
    assert [owner["owner"] for owner in owners] == [0, 1]
    assert owners[0]["baseline"]["resident_vertices"] != owners[1]["baseline"]["resident_vertices"]
    for owner in owners:
        for phase in ("refined", "baseline", "recovered"):
            assert owner[phase]["complete"] and owner[phase]["certified"]
        assert owner["baseline"]["mesh"] == owner["recovered"]["mesh"]
        assert owner["baseline"]["cells"] == owner["recovered"]["cells"]
    for mode in _OWNER_COMMIT_FAULTS:
        faults = [owner["faults"][mode] for owner in owners]
        assert faults[0]["category"] == faults[1]["category"], mode
        assert faults[0]["trace"] == faults[1]["trace"], mode
        assert all(fault["refusals"] == 1 and fault["after_refusal"] == 0 for fault in faults), mode
        assert [fault["injected"] for fault in faults] == [failed_owner == 0, failed_owner == 1], mode
        expected = (
            "CoordinateEnclosureResourceError"
            if mode == "resource"
            else "RuntimeError"
            if mode == "unexpected_lineage"
            else "ValueError"
        )
        assert faults[failed_owner]["cause"] == expected, mode
        assert faults[1 - failed_owner]["cause"] == "NoneType", mode
        category = (
            phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
            if mode == "resource"
            else phx.meshing.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
            if mode == "unexpected_lineage"
            else phx.meshing.MeshingFailureCategory.LINEAGE_FAILED
        )
        assert faults[0]["category"] == category.value, mode
