#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import phydrax as phx


def _run_distributed(scenario: str, *, semantic_owners: bool = False) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", _DISTRIBUTED_SCRIPT, scenario, str(int(semantic_owners))],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        pytest.fail(
            f"Native distributed scenario {scenario!r} exited {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}",
            pytrace=False,
        )


@pytest.mark.parametrize("scenario", ("publication", "coarsen_publication"))
def test_native_owner_local_publication_transfer_fe_fv_and_atomic_rebind(
    scenario: str,
) -> None:
    _run_distributed(scenario)


def test_native_publication_preserves_scientific_markers_and_zone_membership() -> None:
    _run_distributed("scientific_receipts")


@pytest.mark.parametrize("scenario", ("packed_inverse", "packed_inverse_tetra"))
def test_packed_uniform_collective_inverse_retains_raw_epoch_and_cold_scientific_source(
    scenario: str,
) -> None:
    _run_distributed(scenario)


@pytest.mark.parametrize("scenario", ("packed_partial", "packed_partial_tetra"))
def test_packed_partial_collective_inverse_preserves_physical_source_and_cold_raw_epoch(
    scenario: str,
) -> None:
    _run_distributed(scenario)


@pytest.mark.parametrize(
    "scenario",
    ("packed_inverse", "packed_inverse_tetra", "packed_partial", "packed_partial_tetra"),
)
def test_packed_inverse_after_actual_solver_placement_separation(scenario: str) -> None:
    _run_distributed(scenario, semantic_owners=True)


def test_retained_native_ownership_cannot_claim_an_unexecuted_partition_algorithm() -> (
    None
):
    _run_distributed("algorithm_relabel")


@pytest.mark.parametrize(
    "scenario",
    (
        "midpoint",
        "rounded_edge",
        "constraint_history",
        "child_coverage",
        "routes",
        "terminal_status",
        "audit_only",
        "halo_depth",
        "halo_overflow",
    ),
)
def test_native_distributed_collective_certificate_and_resource_boundaries(
    scenario: str,
) -> None:
    _run_distributed(scenario)


def test_historical_shared_vertex_routes_replay_on_changed_device_count() -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import jax
            import jax.numpy as jnp
            import numpy as np
            from phydrax.meshing._restart_distribution import _logical_restart_routes

            assert jax.device_count() == 1
            saved_ids = jnp.asarray(
                [[90, 10, -1, -1], [70, 90, -1, -1], [10, 80, 70, -1]],
                dtype=jnp.int64)
            np.testing.assert_array_equal(
                _logical_restart_routes(saved_ids),
                [[False, True, True], [True, False, True], [True, True, False]])
            # Invalid slots do not create a peer; unrelated stable IDs cannot
            # claim the shared-vertex route that the historical proof declares.
            changed = saved_ids.at[2, 2].set(-1)
            np.testing.assert_array_equal(
                _logical_restart_routes(changed),
                [[False, True, True], [True, False, False], [True, False, False]])
        """),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            f"Historical shared-ID route replay failed:\n{completed.stdout}\n{completed.stderr}",
            pytrace=False,
        )


def test_one_owner_halo_preserves_local_field_without_peer_packets() -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import jax
            import jax.numpy as jnp
            import numpy as np
            from phydrax._fingerprint import canonical_fingerprint
            from phydrax.discretization._cell_mesh import CellMesh, CellMeshStorage
            from phydrax.discretization._distributed_field import (
                DistributedHaloPlan, prepare_owner_local_halo_plans,
            )

            points = np.asarray([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float64)
            cells = np.asarray([[0, 1, 2]], dtype=np.int32)
            dense = CellMesh.from_triangles(points, cells)
            ids = tuple(np.asarray(entities.entity_ids, dtype=np.int64)
                        for entities in dense.topology.entity_sets)
            counts = tuple(values.size for values in ids)
            owners = tuple(np.zeros(values.shape, dtype=np.int32) for values in ids)
            coverage = canonical_fingerprint({
                "topology": dense.topology_id, "entity_ids": ids, "owners": owners})
            storage = CellMeshStorage(
                counts, ids, owners, partition_index=0, partition_count=1,
                logical_topology_id=dense.topology_id,
                logical_geometry_id=dense.geometry_layout_id,
                evidence_id=coverage,
                logical_arrays=(
                    ("cell_vertices", jnp.asarray(cells, dtype=jnp.int64)),
                    ("coordinates", jnp.asarray(points, dtype=jnp.float64)),
                    ("vertex_global_ids", jnp.asarray(ids[0], dtype=jnp.int64))),
                local_coordinates=points, local_blocks=dense.blocks,
                local_physical_boundary_facets=np.ones(counts[1], dtype=np.bool_),
                local_neighborhood_complete=np.ones(counts[2], dtype=np.bool_),
            )
            mesh = CellMesh(points, dense.blocks, storage=storage)
            halo, = prepare_owner_local_halo_plans(
                (mesh,), 2, devices=tuple(jax.devices()), axis_name="parts", message_capacity=1)
            np.testing.assert_array_equal(halo.local_global_ids, ids[2])
            np.testing.assert_array_equal(halo.local_owned, [True])
            value = jnp.asarray([2.75], dtype=jnp.float64)
            def exchange(plan: DistributedHaloPlan, field: jax.Array) -> jax.Array:
                return plan.exchange(field, plan.partition_index, axis_name="parts")
            np.testing.assert_array_equal(jax.jit(exchange)(halo, value), value)
            assert halo.permutations == ()
        """),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            f"One-owner halo consumer failed:\n{completed.stdout}\n{completed.stderr}",
            pytrace=False,
        )


_DISTRIBUTED_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import tempfile
    from typing import Any

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    import phydrax as phx
    import os
    import json
    from phydrax._meshcore import meshcore_runtime_identity
    print("NATIVE_RUNTIME_IDENTITY", os.getpid(), json.dumps(meshcore_runtime_identity(), sort_keys=True), flush=True)
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._adaptive_simplex import (
        AdaptiveSimplexState, coarsen_adaptive_simplex_parts, refine_adaptive_simplex_parts,
    )
    from phydrax.discretization.fem._distributed import prepare_owner_local_finite_element
    from phydrax.discretization.fem._generic import FiniteElementPlan, FiniteElementFieldSpec
    from phydrax.discretization.fem._reference import lagrange_element
    from phydrax.meshing._distribution import _compiled_bisection_checks, expand_partitioned_simplex_neighborhood
    from phydrax.meshing.providers._native_publication import NativeCertificationRequest, publish_native_result
    from phydrax.linalg._distributed import DistributedKrylovPolicy
    from phydrax.discretization._distributed_field import prepare_owner_local_halo_plans
    from phydrax.discretization.finite_volume._unstructured_dynamics import (
        PreparedUnstructuredFiniteVolumeDynamics, execute_owner_local_finite_volume_advance,
    )
    from phydrax.lifecycle._array_artifact import ArrayArtifactProvenance, write_typed_array_artifact, read_typed_array_artifact
    from phydrax.lifecycle._distributed_checkpoint import snapshot_addressable_arrays

    M = phx.meshing
    D = phx.discretization
    scenario = sys.argv[1]
    semantic_owners = len(sys.argv) > 2 and sys.argv[2] == "1"
    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    if scenario == "rounded_edge":
        points = np.asarray(((0.1, 0.2), (0.4, 0.2), (0.3, 0.7), (0.0, 0.7)), dtype=np.float64)
    if scenario == "halo_depth":
        points = np.asarray(((0, 0), (0.5, 0), (1, 0), (1, 1), (0.5, 1), (0, 1)), dtype=np.float64)
        cells = np.asarray(((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)), dtype=np.int32)
    if scenario == "packed_inverse":
        points = np.asarray(((0., 0.), (2., 0.), (1., .5), (1., -3.)), dtype=np.float64)
        cells = np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
    if scenario == "packed_inverse_tetra":
        points = np.asarray(((0., 0., 0.), (2., 0., 0.), (1., .5, 0.), (1., .2, 1.), (1., -3., -1.)), dtype=np.float64)
        cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
    if scenario in ("packed_partial", "packed_partial_tetra"):
        dimension = 3 if scenario.endswith("tetra") else 2
        simplex = np.concatenate((np.zeros((1, dimension), dtype=np.float64), np.eye(dimension, dtype=np.float64)))
        if dimension == 2:
            component = np.asarray(((0., 0.), (2., 0.), (1., .5), (1., -3.)), dtype=np.float64)
            component_cells = np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32)
        else:
            component = np.asarray(((0., 0., 0.), (2., 0., 0.), (1., .5, 0.), (1., .2, 1.), (1., -3., -1.)), dtype=np.float64)
            component_cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32)
        points = np.concatenate((component, simplex + 4.0))
        cells = np.concatenate((component_cells, np.arange(component.shape[0], points.shape[0], dtype=np.int32)[None]))
    mesh = D.CellMesh.from_tetrahedra(points, cells) if scenario in ("packed_inverse_tetra", "packed_partial_tetra") else D.CellMesh.from_triangles(points, cells)
    seed = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    loop = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    if scenario == "halo_depth":
        loop = np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)), dtype=np.int32)
    if scenario == "packed_inverse":
        loop = np.asarray(((0, 3), (3, 1), (1, 2), (2, 0)), dtype=np.int32)
    if scenario == "packed_partial":
        loop = np.asarray(((0, 3), (3, 1), (1, 2), (2, 0), (4, 5), (5, 6), (6, 4)), dtype=np.int32)
    if scenario in ("packed_inverse_tetra", "packed_partial_tetra"):
        faces = cells[:, np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)].reshape((-1, 3))
        _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
        loop = faces[first[counts == 1]]
    source_id = "packed-original" if scenario.startswith("packed_") else "unit-square"
    domain = phx.geometry.PiecewiseLinearDomain(
        points, loop, np.tile(np.asarray((0, -1), dtype=np.int32), (loop.shape[0], 1)),
        ("domain",), source_id=source_id,
    )
    source_attributes, source_zones = (), ()
    if scenario == "scientific_receipts" or scenario.startswith("packed_"):
        cell_entities = mesh.entity_set(mesh.topological_dimension)
        all_cells = M.MeshingScope(
            mesh.mesh_id, mesh.numeric_version, M.MeshingEntityKind.MESH,
            mesh.topological_dimension, cell_entities.entity_set_id, np.asarray(cell_entities.entity_ids, dtype=np.int64),
        )
        source_attributes = (M.MeshAttribute(
            "marker", M.MeshAttributeRole.MARKER, all_cells,
            np.asarray((19, 23, 31) if scenario in ("packed_partial", "packed_partial_tetra") else (19, 23), dtype=np.int64),
        ),)
        source_zones = tuple(
            M.MeshZone(f"cell:{index}", M.MeshZoneRole.USER, M.MeshingScope(
                mesh.mesh_id, mesh.numeric_version, M.MeshingEntityKind.MESH,
                mesh.topological_dimension, cell_entities.entity_set_id, np.asarray((identifier,), dtype=np.int64),
            ))
            for index, identifier in enumerate(np.asarray(cell_entities.entity_ids, dtype=np.int64))
        )
    source = publish_native_result(
        seed.mesh, seed.coordinate_contract, M.MeshingComplianceReport(source_id), (),
        seed.provider, {"kind": "independent-domain-certified-source"},
        NativeCertificationRequest(
            M.MeshCertificationSchedule("volume_plc"), source_id, "r0", M.MeshingLimits(),
            domain=domain, cell_regions=np.zeros(cells.shape[0], dtype=np.int32),
        ),
        audit_policy=M.CellMeshAuditPolicy(watertight_boundary=M.CellMeshAuditDisposition.REJECT),
        derivative_mode=M.MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
        enforced_limits=(), unenforced_limits=(),
        attributes=source_attributes, zones=source_zones,
    )
    if scenario.startswith("packed_"):
        from phydrax._array_archive import ArrayArchiveLimits
        from phydrax.lifecycle._meshing_sources import write_meshing_source_closure
        from phydrax.discretization._cell_geometry_validity import cell_geometry_id
        from phydrax.discretization._cell_mesh import CellMeshStorage
        import subprocess
        import os
        original_coefficients = source.geometry.source_coordinates()
        fine = M.execute_mesh_adaptation(M.prepare_mesh_adaptation(
            source, M.MarkedMeshAdaptation(), policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=M.BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
        ))
        lineage = fine.hierarchy.uniform_refinement
        assert lineage is not None and lineage.child_ids.shape[1] == (24 if scenario.endswith("tetra") else 6)
        assert lineage.source is source
        fine_ids = np.concatenate(tuple(np.asarray(block.global_ids) for block in fine.target.mesh.blocks))
        owners = np.full(fine_ids.shape, -1, dtype=np.int32)
        for rank, children in enumerate(np.asarray(lineage.child_ids)):
            owners[np.isin(fine_ids, children)] = (0 if rank < 2 else 1) if scenario in ("packed_partial", "packed_partial_tetra") else rank
        assert np.all(owners >= 0)
        # This scenario retains authored provider routes, not balanced repartition.
        partition_policy = M.MeshPartitionPolicy(
            M.MeshPartitionKind.PROVIDER, 2, halo_width=2, maximum_imbalance=2.0,
        )
        distribution = M.prepare_mesh_distribution(
            M.MeshPart("domain", fine.target), policy=partition_policy, ownership=owners,
        )
        policy = M.MeshAdaptationPolicy(
            M.MeshAdaptationRoute.DEVICE_BISECTION,
            device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=64, cell_capacity=64),
            distribution=distribution, partition_policy=partition_policy,
        )
        prepared = M.prepare_adaptive_simplex(fine.target, policy=policy, hierarchy=fine.hierarchy)
        partitioned = M.partition_adaptive_simplex(prepared)
        # A real protected preparation must refuse removal of an issued source
        # corner without changing either the submitted raw epoch or its source.
        from phydrax.meshing._device_adaptation import prepare_partitioned_mesh_adaptation
        born = np.setdiff1d(fine.target.mesh.vertex_global_ids, source.mesh.vertex_global_ids)
        protected_scope = M.MeshingScope(
            fine.target.mesh.mesh_id, fine.target.mesh.numeric_version,
            M.MeshingEntityKind.MESH, 0, fine.target.mesh.entity_set(0).entity_set_id, born[:1],
        )
        protected_policy = M.MeshAdaptationPolicy(
            M.MeshAdaptationRoute.DEVICE_BISECTION, device_policy=policy.device_policy,
            distribution=distribution, partition_policy=partition_policy,
            protected_scopes=(protected_scope,),
        )
        protected = prepare_partitioned_mesh_adaptation(M.prepare_mesh_adaptation(
            fine.target, M.MarkedMeshAdaptation(
                (), np.sort(fine_ids), hierarchy=fine.hierarchy,
            ), policy=protected_policy,
        ))
        protected_raw = coarsen_adaptive_simplex_parts(
            protected.layout, protected.parts, protected.states, protected.states.mesh.cell_active,
        ).state
        protected_before = tuple(np.array(value, copy=True) for value in jax.tree_util.tree_leaves(protected_raw))
        protected_results = M.commit_partitioned_adaptive_simplex(protected, protected_raw)
        protected_banks = dict(protected_results[0].target.collective_evidence.logical_arrays)
        protected_position = np.flatnonzero(protected_banks["vertex_global_ids"] == born[0])
        assert protected_position.size == 1
        fine_position = np.flatnonzero(np.asarray(fine.target.mesh.vertex_global_ids) == born[0])
        np.testing.assert_array_equal(
            protected_banks["coordinates"][protected_position[0]],
            np.asarray(fine.target.mesh.coordinates)[fine_position[0]],
        )
        assert any(result.hierarchy.uniform_refinement is not None for result in protected_results)
        for value, original in zip(jax.tree_util.tree_leaves(protected_raw), protected_before, strict=True):
            np.testing.assert_array_equal(value, original)
        if semantic_owners:
            # Execute a real sparse binary refinement, publish it, then use the
            # accepted restart owner transport. One binary family is physically
            # coalesced while its active scientific solver owners remain mixed.
            from phydrax._execution_runtime import ExecutionRuntime
            from phydrax.meshing._restart_distribution import repack_simplex_restart
            from phydrax.meshing._device_adaptation import commit_partitioned_mesh_restart
            sparse = partitioned.cell_marks(np.asarray(lineage.child_ids)[0, :1])
            split = refine_adaptive_simplex_parts(
                partitioned.layout, partitioned.parts, partitioned.states, sparse,
            ).state
            split_results = M.commit_partitioned_adaptive_simplex(partitioned, split)
            accepted = split_results[0].target
            rank = jnp.arange(2, dtype=jnp.int32)[:, None]
            physical = jnp.where(split.mesh.cell_ids >= 0, rank, -1)
            vertex_owners = jnp.full(split.mesh.vertex_ids.shape, 2, dtype=jnp.int32)
            for owner in range(2):
                present = jnp.any(
                    split.mesh.vertex_ids[:, :, None] == split.mesh.vertex_ids[owner, None, None, :],
                    axis=-1,
                )
                vertex_owners = jnp.minimum(vertex_owners, jnp.where(present, owner, 2))
            vertex_owners = jnp.where(split.mesh.vertex_ids >= 0, vertex_owners, -1)
            proposal = jnp.broadcast_to(rank, split.mesh.cell_ids.shape)
            binary = np.flatnonzero(
                np.all(np.asarray(split.children[0]) >= 0, axis=1)
                & np.all(np.asarray(split.mesh.cell_active[0])[np.maximum(np.asarray(split.children[0]), 0)], axis=1),
            )
            assert binary.size > 0
            second = int(np.asarray(split.children[0])[binary[0], 1])
            proposal = proposal.at[0, second].set(1)
            repack = repack_simplex_restart(
                ExecutionRuntime.current().root_group, partitioned.layout,
                split, physical, vertex_owners, proposal, source_result=accepted,
            )
            np.testing.assert_array_equal(repack.status, 0)
            assert np.any(
                np.asarray(repack.states.mesh.cell_active)
                & (np.asarray(repack.solver_cell_owners) != np.arange(2)[:, None]),
            )
            accepted_distribution = M.prepare_mesh_distribution(
                M.MeshPart("domain", accepted), policy=partition_policy,
            )
            accepted_policy = M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.DEVICE_BISECTION,
                device_policy=policy.device_policy, distribution=accepted_distribution,
                partition_policy=partition_policy,
            )
            placed_results = commit_partitioned_mesh_restart(
                repack, M.prepare_mesh_adaptation(
                    accepted, M.MarkedMeshAdaptation(hierarchy=split_results[0].hierarchy), policy=accepted_policy,
                ),
            )
            placed = placed_results[0].target
            placed_distribution = M.prepare_mesh_distribution(
                M.MeshPart("domain", placed), policy=partition_policy,
            )
            policy = M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.DEVICE_BISECTION,
                device_policy=policy.device_policy, distribution=placed_distribution,
                partition_policy=partition_policy,
            )
            partitioned = prepare_partitioned_mesh_adaptation(M.prepare_mesh_adaptation(
                placed, M.MarkedMeshAdaptation(hierarchy=placed_results[0].hierarchy), policy=policy,
            ))
            assert np.any(
                np.asarray(partitioned.states.mesh.cell_active)
                & (np.asarray(partitioned.solver_cell_owners) != np.arange(2)[:, None]),
            )
        coarsen_marks = partitioned.states.mesh.cell_active
        partial = scenario in ("packed_partial", "packed_partial_tetra")
        if partial:
            coarsen_marks = coarsen_marks & (jnp.arange(2)[:, None] == 0)
        compiled = coarsen_adaptive_simplex_parts(
            partitioned.layout, partitioned.parts, partitioned.states, coarsen_marks,
        ).state
        before = tuple(np.array(value, copy=True) for value in jax.tree_util.tree_leaves(compiled))
        results = M.commit_partitioned_adaptive_simplex(partitioned, compiled)
        evidence = results[0].target.collective_evidence
        # Receipt rebinding may reconstruct immutable PyTree containers, but it
        # must retain every submitted numerical owner without copying.
        assert all(
            retained is submitted
            for retained, submitted in zip(
                jax.tree_util.tree_leaves(evidence.compiled_states),
                jax.tree_util.tree_leaves(compiled),
                strict=True,
            )
        )
        assert evidence.uniform_refinement.lineage_id == lineage.lineage_id
        assert all(
            retained is submitted
            for retained, submitted in zip(
                jax.tree_util.tree_leaves(evidence.uniform_refinement),
                jax.tree_util.tree_leaves(lineage),
                strict=True,
            )
        )
        for value, original in zip(jax.tree_util.tree_leaves(compiled), before, strict=True):
            np.testing.assert_array_equal(value, original)
        # Refusal is atomic for the actual commit consumer, not only for a
        # detached evidence validator.
        invalid_compiled = eqx.tree_at(
            lambda value: value.counters, compiled, -jnp.ones_like(compiled.counters),
        )
        try:
            M.commit_partitioned_adaptive_simplex(partitioned, invalid_compiled)
        except (ValueError, M.MeshingFailure):
            pass
        else:
            raise AssertionError("A negative raw execution inventory was published.")
        for value, original in zip(jax.tree_util.tree_leaves(compiled), before, strict=True):
            np.testing.assert_array_equal(value, original)
        banks = dict(evidence.logical_arrays)
        original_ids = np.concatenate(tuple(np.asarray(block.global_ids) for block in source.mesh.blocks))
        expected_cells = 2 + lineage.child_ids.shape[1] if partial else 2
        assert evidence.global_entity_counts[-1] == expected_cells
        raw_ids = np.asarray(partitioned.states.mesh.cell_ids)
        raw_owners = np.asarray(partitioned.solver_cell_owners)
        raw_active = np.asarray(partitioned.states.mesh.cell_active)
        raw_children = np.asarray(partitioned.states.children)
        published_owners = np.asarray(banks["cell_owners"])
        for parent, siblings in zip(np.asarray(lineage.parent_ids), np.asarray(lineage.child_ids), strict=True):
            published = np.flatnonzero(np.asarray(banks["cell_global_ids"]) == parent)
            if published.size:
                support = []
                pending = [tuple(position) for position in np.argwhere(np.isin(raw_ids, siblings))]
                assert len(pending) == siblings.size
                visited = set()
                while pending:
                    owner, slot = pending.pop()
                    assert (owner, slot) not in visited
                    visited.add((owner, slot))
                    if raw_active[owner, slot]:
                        support.append(raw_owners[owner, slot])
                    else:
                        children = raw_children[owner, slot]
                        assert np.all(children >= 0)
                        pending.extend((owner, int(child)) for child in children)
                assert support and np.all(np.asarray(support) >= 0)
                assert published_owners[published[0]] == np.min(support)
        if not partial:
            np.testing.assert_array_equal(banks["cell_global_ids"][:2], original_ids)
            vertex_count = source.mesh.vertex_global_ids.size
            np.testing.assert_array_equal(banks["vertex_global_ids"][:vertex_count], source.mesh.vertex_global_ids)
            np.testing.assert_array_equal(banks["coordinates"][:vertex_count], source.mesh.coordinates)
            assert evidence.coordinate_geometry_id == cell_geometry_id(source.geometry)
        assert not np.array_equal(banks["epoch/cell_ids"], compiled.mesh.cell_ids)
        assert any(result.hierarchy.uniform_refinement is not None for result in results) == partial
        authored = dict(zip(
            np.asarray(source_attributes[0].scope.global_entity_ids),
            np.asarray(source_attributes[0].global_values), strict=True,
        ))
        original_parent = {int(parent): int(parent) for parent in lineage.parent_ids}
        original_parent.update({
            int(child): int(parent)
            for parent, siblings in zip(lineage.parent_ids, lineage.child_ids, strict=True)
            for child in siblings
        })
        accepted_ids = np.asarray(banks["cell_global_ids"])[:expected_cells]
        expected_markers = np.asarray([authored[original_parent[int(identifier)]] for identifier in accepted_ids])
        for result in results:
            from fractions import Fraction
            from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
            marker = result.target.attributes[0]
            np.testing.assert_array_equal(marker.scope.global_entity_ids, accepted_ids)
            np.testing.assert_array_equal(marker.global_values, expected_markers)
            local_rows = np.searchsorted(accepted_ids, np.asarray(marker.scope.entity_ids))
            np.testing.assert_array_equal(marker.values, expected_markers[local_rows])
            assert tuple(zone.name for zone in result.target.zones) == tuple(zone.name for zone in source_zones)
            for zone, original_zone in zip(result.target.zones, source_zones, strict=True):
                members = np.asarray(original_zone.scope.global_entity_ids)
                expected = accepted_ids[np.isin(
                    np.asarray([original_parent[int(identifier)] for identifier in accepted_ids]), members,
                )]
                np.testing.assert_array_equal(zone.scope.global_entity_ids, expected)
            storage = result.target.mesh.storage
            assert isinstance(storage, CellMeshStorage)
            assert result.target.geometry.source_coordinates() == source.geometry.source_coordinates(
                storage.coordinate_global_ids,
            )
            assert lineage.source is source
            assert source.geometry.source_coordinates() == original_coefficients
            elements, routes, coefficients = result.target.geometry.resolve(result.target.mesh)
            for block, element, route in zip(result.target.mesh.blocks, elements, routes, strict=True):
                action = np.asarray(element.barycentric_weights) if isinstance(element, BarycentricCellGeometryElement) else np.eye(block.vertices.shape[1], dtype=np.float64)
                for row, source_route in zip(np.asarray(block.vertices), np.asarray(route), strict=True):
                    exact = tuple(tuple(sum(
                        (Fraction(float(weight)) * Fraction(float(coefficients[index, axis]))
                         for index, weight in zip(source_route, corner, strict=True)), Fraction(0),
                    ) for axis in range(points.shape[1])) for corner in action)
                    np.testing.assert_array_equal(
                        result.target.mesh.coordinates[row], np.asarray(exact, dtype=np.float64),
                    )
            assert result.target.execution_evidence is not None
            assert int(result.target.execution_evidence.externally_charged_work) > 0
        from phydrax.meshing._device_adaptation import validate_partitioned_mesh_evidence
        altered_raw = eqx.tree_at(
            lambda value: value.compiled_states.counters, evidence,
            evidence.compiled_states.counters.at[0, 0].add(1),
        )
        try:
            validate_partitioned_mesh_evidence(results[0].target.mesh.storage, altered_raw)
        except ValueError:
            pass
        else:
            raise AssertionError("A changed raw compiled execution record was accepted.")
        with tempfile.TemporaryDirectory() as directory:
            limits = ArrayArchiveLimits()
            assert (limits.max_members, limits.max_manifest_nesting, limits.max_array_rank) == (257, 16, 8)
            receipt = write_meshing_source_closure(
                Path(directory) / "packed-collective", (results[0].target, policy, results[0].hierarchy), limits=limits,
            )
            environment = dict(os.environ)
            environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
            subprocess.run([sys.executable, "-c", '''
    import sys
    import numpy as np
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
    from phydrax.meshing._device_adaptation import validate_partitioned_mesh_evidence
    import os
    import json
    from phydrax._meshcore import meshcore_runtime_identity
    print("NATIVE_RUNTIME_IDENTITY", os.getpid(), json.dumps(meshcore_runtime_identity(), sort_keys=True), flush=True)
    target, policy, hierarchy = read_meshing_source_closure(
        sys.argv[1], expected_content_id=sys.argv[2], limits=ArrayArchiveLimits())
    assert policy.policy_id == sys.argv[3]
    proof = target.collective_evidence
    validate_partitioned_mesh_evidence(target.mesh.storage, proof)
    assert proof.uniform_refinement.source.result_id == sys.argv[4]
    assert proof.global_entity_counts[-1] == int(sys.argv[5])
    assert not np.array_equal(dict(proof.logical_arrays)["epoch/cell_ids"], proof.compiled_states.mesh.cell_ids)
    # A changed execution count is not permission to discard allocated history
    # or enlarge the original per-part capacity.
    import jax
    import jax.numpy as jnp
    import phydrax as phx
    from phydrax._execution_runtime import ExecutionRuntime
    from phydrax.meshing._device_adaptation import _retained_partition_states, commit_partitioned_mesh_restart
    from phydrax.meshing._restart_distribution import repack_simplex_restart
    layout, saved, exterior = _retained_partition_states(target)
    if np.count_nonzero(np.asarray(saved.mesh.cell_ids) >= 0) > layout.cell_capacity:
        before = tuple(np.array(value, copy=True) for value in jax.tree_util.tree_leaves(saved))
        ranks = jnp.arange(saved.mesh.cell_ids.shape[0], dtype=jnp.int32)[:, None]
        cell_owners = jnp.where(saved.mesh.cell_ids >= 0, ranks, -1)
        vertex_owners = jnp.full(saved.mesh.vertex_ids.shape, saved.mesh.vertex_ids.shape[0], dtype=jnp.int32)
        for owner in range(saved.mesh.vertex_ids.shape[0]):
            present = jnp.any(saved.mesh.vertex_ids[:, :, None] == saved.mesh.vertex_ids[owner, None, None, :], axis=-1)
            vertex_owners = jnp.minimum(vertex_owners, jnp.where(present, owner, saved.mesh.vertex_ids.shape[0]))
        vertex_owners = jnp.where(saved.mesh.vertex_ids >= 0, vertex_owners, -1)
        repack = repack_simplex_restart(
            ExecutionRuntime.current().root_group, layout, saved, cell_owners, vertex_owners,
            jnp.zeros_like(cell_owners), source_result=target,
        )
        assert np.all(np.asarray(repack.status) & int(phx.discretization.AdaptiveSimplexStatus.CAPACITY_EXCEEDED))
        source_partition = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.PROVIDER, 2, halo_width=2)
        distribution = phx.meshing.prepare_mesh_distribution(phx.meshing.MeshPart("domain", target), policy=source_partition)
        restart_policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION, device_policy=policy.device_policy,
            distribution=distribution,
            partition_policy=phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.PROVIDER, 1, halo_width=2),
        )
        try:
            commit_partitioned_mesh_restart(repack, phx.meshing.prepare_mesh_adaptation(
                target, phx.meshing.MarkedMeshAdaptation(hierarchy=hierarchy), policy=restart_policy))
        except ValueError:
            pass
        else:
            raise AssertionError("An over-capacity changed-device raw inventory was published.")
        for value, original in zip(jax.tree_util.tree_leaves(saved), before, strict=True):
            np.testing.assert_array_equal(value, original)
    ''', str(receipt.path), receipt.content_id, policy.policy_id, source.result_id, str(expected_cells)],
                env=environment, check=True)
            environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
            subprocess.run([sys.executable, "-c", '''
    import sys
    import numpy as np
    import phydrax as phx
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.discretization._cell_mesh import CellMeshStorage
    from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
    import os
    import json
    from phydrax._meshcore import meshcore_runtime_identity
    print("NATIVE_RUNTIME_IDENTITY", os.getpid(), json.dumps(meshcore_runtime_identity(), sort_keys=True), flush=True)
    target, saved_policy, hierarchy = read_meshing_source_closure(
        sys.argv[1], expected_content_id=sys.argv[2], limits=ArrayArchiveLimits())
    M = phx.meshing
    # Cold continuation preserves the saved provider routes across refinement.
    partition_policy = M.MeshPartitionPolicy(
        M.MeshPartitionKind.PROVIDER, 2, halo_width=2, maximum_imbalance=2.0,
    )
    distribution = M.prepare_mesh_distribution(M.MeshPart("domain", target), policy=partition_policy)
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION, device_policy=saved_policy.device_policy,
        distribution=distribution, partition_policy=partition_policy,
    )
    ids = np.asarray(target.collective_evidence.entity_ids[-1])[:target.collective_evidence.global_entity_counts[-1]]
    original = target.collective_evidence.uniform_refinement.source
    original_coefficients = original.geometry.source_coordinates()
    continued = M.execute_mesh_adaptation(M.prepare_mesh_adaptation(
        target, M.MarkedMeshAdaptation(ids[:1], hierarchy=hierarchy), policy=policy,
    ))
    assert continued.status is M.MeshAdaptationStatus.COMPLETE
    continued.target.audit.require_passed()
    storage = continued.target.mesh.storage
    assert isinstance(storage, CellMeshStorage)
    assert continued.target.geometry.source_coordinates() == original.geometry.source_coordinates(
        storage.coordinate_global_ids,
    )
    assert target.collective_evidence.uniform_refinement.source is original
    assert original.geometry.source_coordinates() == original_coefficients
    ''', str(receipt.path), receipt.content_id], env=environment, check=True)
        print("PACKED: actual raw epoch retained; original collective physical field restored; cold replay on one device")
        sys.exit(0)
    if scenario == "audit_only":
        source = seed
    kind = M.MeshPartitionKind.PROVIDER if scenario == "halo_depth" else M.MeshPartitionKind.MORTON
    partition_policy = M.MeshPartitionPolicy(kind, 2, halo_width=2)
    distribution = M.prepare_mesh_distribution(
        M.MeshPart("domain", source), policy=partition_policy,
        ownership=np.asarray((0, 0, 1, 1), dtype=np.int32) if scenario == "halo_depth" else None,
    )
    if scenario == "algorithm_relabel":
        partition_policy = M.MeshPartitionPolicy(M.MeshPartitionKind.HILBERT, 2, halo_width=2)
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=64, cell_capacity=64),
        distribution=distribution, partition_policy=partition_policy,
        audit_policy=M.CellMeshAuditPolicy(watertight_boundary=M.CellMeshAuditDisposition.REJECT),
    )
    prepared = M.prepare_adaptive_simplex(source, policy=policy)
    partitioned = M.partition_adaptive_simplex(prepared)
    marks = partitioned.cell_marks(np.asarray((0,), dtype=np.int64))
    update = D.refine_adaptive_simplex_parts(partitioned.layout, partitioned.parts, partitioned.states, marks)
    states = update.state
    if scenario == "coarsen_publication":
        from phydrax.discretization._adaptive_simplex import coarsen_adaptive_simplex_parts
        states = coarsen_adaptive_simplex_parts(
            partitioned.layout, partitioned.parts, states, states.mesh.cell_active
        ).state

    if scenario == "midpoint":
        issued = np.flatnonzero(np.asarray(states.mesh.vertex_ids[0]) >= points.shape[0])
        issued = issued[np.asarray(states.mesh.vertex_active[0])[issued]]
        coordinates = states.mesh.coordinates.at[0, issued[0], 1].add(0.125)
        states = eqx.tree_at(lambda value: value.mesh.coordinates, states, coordinates)
        checks = _compiled_bisection_checks(partitioned.parts, states, partitioned.states, partitioned.source_exterior)
        assert not np.all(np.asarray(checks[:, 2]))
    elif scenario == "constraint_history":
        issued = np.flatnonzero((np.asarray(states.mesh.vertex_ids[0]) >= points.shape[0])
                                & np.asarray(states.mesh.vertex_active[0]))
        endpoints = np.asarray(states.vertex_parents[0, issued[0]], dtype=np.int64)
        code = np.min(endpoints) * partitioned.layout.vertex_capacity + np.max(endpoints)
        protected = partitioned.states.protected_codes.at[0, 0].set(code)
        initial = eqx.tree_at(lambda value: value.protected_codes, partitioned.states, protected)
        changed = eqx.tree_at(lambda value: value.protected_codes, states, protected)
        assert np.all(np.asarray(changed.status_flags) == 0)
        checks = _compiled_bisection_checks(partitioned.parts, changed, initial, partitioned.source_exterior)
        assert not np.all(np.asarray(checks[:, 2]))
    elif scenario == "child_coverage":
        split = np.flatnonzero(np.asarray(states.children[0, :, 0]) >= 0)[0]
        children = states.children.at[0, split, 0].set(states.children[0, split, 1])
        states = eqx.tree_at(lambda value: value.children, states, children)
        checks = _compiled_bisection_checks(partitioned.parts, states, partitioned.states, partitioned.source_exterior)
        assert not np.all(np.asarray(checks[:, 1]))
    elif scenario == "routes":
        parts = D.AdaptiveSimplexParts(tuple(jax.devices()), neighbor_pairs=())
        checks = _compiled_bisection_checks(parts, states, partitioned.states, partitioned.source_exterior)
        assert not np.all(np.asarray(checks[:, 4]))
    elif scenario == "rounded_edge":
        # Exact coefficient actions are the geometry; float midpoints are their
        # rounding, witnessed by retained native residual signs.
        from phydrax.discretization._cell_geometry import BarycentricCellGeometryElement
        from phydrax.meshing._result import CollectiveMeshEvidence

        results = M.commit_partitioned_adaptive_simplex(partitioned, states)
        assert len(results) == 2
        issued = (np.asarray(states.mesh.vertex_ids) >= points.shape[0]) & np.asarray(states.mesh.vertex_active)
        rounded = np.asarray(states.mesh.coordinates)[issued]
        assert rounded.size
        for result in results:
            evidence = result.target.collective_evidence
            assert isinstance(evidence, CollectiveMeshEvidence)
            receipts = dict(evidence.logical_arrays)
            valid = np.asarray(receipts["native/midpoint_valid"])
            signs = np.asarray(receipts["native/midpoint_signs"])
            assert np.array_equal(valid, issued)
            assert np.all(signs[valid] != 0), "An off-edge rounded midpoint hid its exact residual"
            geometry = result.target.geometry
            assert geometry.restriction_source is not None
            for element in geometry.elements:
                assert isinstance(element, BarycentricCellGeometryElement)
                weights = np.asarray(element.barycentric_weights)
                assert np.array_equal(np.round(2.0 * weights), 2.0 * weights)
                np.testing.assert_array_equal(np.sum(weights, axis=1), 1.0)
            coefficients = np.asarray(geometry.coordinates)
            assert all(np.any(np.all(points == row, axis=1)) for row in coefficients)
            assert not any(np.any(np.all(coefficients == row, axis=1)) for row in rounded)
        slot = np.flatnonzero(issued[0])[0]
        value = states.mesh.coordinates[0, slot, 1]
        coordinates = states.mesh.coordinates.at[0, slot, 1].set(jnp.nextafter(value, jnp.inf))
        perturbed = eqx.tree_at(lambda value: value.mesh.coordinates, states, coordinates)
        try:
            M.commit_partitioned_adaptive_simplex(partitioned, perturbed)
        except M.MeshingFailure as error:
            assert error.category is M.MeshingFailureCategory.QUALITY_REJECTED
            assert error.stage == "distributed-reference-restriction"
        else:
            raise AssertionError("A vertex off its declared parent-edge sample claimed exact restriction")
    elif scenario == "terminal_status":
        clocks = states.clocks.at[1, 2].set(int(D.AdaptiveSimplexStatus.CAPACITY_EXCEEDED))
        states = eqx.tree_at(lambda value: value.clocks, states, clocks)
        try:
            M.commit_partitioned_adaptive_simplex(partitioned, states)
        except M.MeshingFailure as error:
            assert error.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
        else:
            raise AssertionError("One-part failure published a distributed epoch")
    elif scenario == "audit_only":
        try:
            M.commit_partitioned_adaptive_simplex(partitioned, states)
        except M.MeshingFailure as error:
            assert error.category is M.MeshingFailureCategory.INVALID_SPECIFICATION
        else:
            raise AssertionError("A local audit was promoted to a global certificate")
    elif scenario == "algorithm_relabel":
        source_result_id = source.result_id
        source_owners = np.asarray(distribution.partition.cell_owner).copy()
        try:
            M.commit_partitioned_adaptive_simplex(partitioned, states)
        except M.MeshingFailure as error:
            assert error.category is M.MeshingFailureCategory.LINEAGE_FAILED
        else:
            raise AssertionError("Unmigrated MORTON owners claimed HILBERT partitioning")
        assert source.result_id == source_result_id
        np.testing.assert_array_equal(distribution.partition.cell_owner, source_owners)
        assert distribution.evidence.kind is M.MeshPartitionKind.MORTON
    elif scenario in ("halo_depth", "halo_overflow"):
        widths = (1, 2, 3) if scenario == "halo_depth" else (2,)
        counts = []
        for width in widths:
            work = expand_partitioned_simplex_neighborhood(
                partitioned.parts, states, partitioned.states, partitioned.source_exterior,
                halo_width=width, cell_capacity=64, vertex_capacity=3 if scenario == "halo_overflow" else 64,
            )
            if scenario == "halo_overflow":
                flags = np.asarray(work.status)
                assert np.all(flags & int(D.AdaptiveSimplexStatus.CAPACITY_EXCEEDED))
                break
            assert not np.any(np.asarray(work.status))
            valid = np.asarray(work.cell_valid)
            counts.append(np.sum(valid, axis=1))
            ids = np.asarray(work.source_vertex_ids)
            weights = np.asarray(work.source_weights)
            references = points[np.maximum(ids, 0)]
            rebuilt = np.sum(np.where((ids >= 0)[..., None], references * weights[..., None], 0), axis=-2)
            np.testing.assert_allclose(np.asarray(work.cell_coordinates)[valid], rebuilt[valid], atol=0, rtol=0)
            np.testing.assert_allclose(np.sum(weights, axis=-1)[valid], 1, atol=0, rtol=0)
        if scenario == "halo_depth":
            assert np.any(counts[1] > counts[0]), (counts[0], counts[1])
            assert np.all(counts[2] >= counts[1])
    else:
        original_get = jax.device_get
        transfers = []
        def observed_get(value: Any) -> Any:
            if isinstance(value, AdaptiveSimplexState):
                assert value.mesh.cells.ndim == 2, "The commit transferred all part states"
                transfers.append(value.mesh.cells.shape[0])
            return original_get(value)
        jax.device_get = observed_get
        try:
            results = M.commit_partitioned_adaptive_simplex(partitioned, states)
        finally:
            jax.device_get = original_get
        assert len(results) == 2
        assert transfers and max(transfers) == partitioned.layout.cell_capacity
        if scenario == "scientific_receipts":
            authored = dict(zip(
                np.asarray(source_attributes[0].scope.global_entity_ids),
                np.asarray(source_attributes[0].global_values), strict=True,
            ))
            for result in results:
                target = result.target
                banks = dict(target.collective_evidence.logical_arrays)
                count = target.mesh.storage.global_entity_counts[2]
                ids = np.asarray(banks["cell_global_ids"])[:count]
                roots = np.asarray(banks["organization/cell/source_ids"])[:count]
                expected = np.asarray([authored[identifier] for identifier in roots], dtype=np.int64)
                marker = target.attributes[0]
                np.testing.assert_array_equal(np.asarray(marker.scope.global_entity_ids), ids)
                np.testing.assert_array_equal(np.asarray(marker.global_values), expected)
                local_rows = np.searchsorted(ids, np.asarray(marker.scope.entity_ids))
                np.testing.assert_array_equal(np.asarray(marker.values), expected[local_rows])
                assert tuple(zone.name for zone in target.zones) == ("cell:0", "cell:1")
                for index, zone in enumerate(target.zones):
                    source_ids = np.asarray(source_zones[index].scope.global_entity_ids)
                    np.testing.assert_array_equal(
                        np.asarray(zone.scope.global_entity_ids), ids[np.isin(roots, source_ids)],
                    )
            sys.exit(0)
        single = D.refine_adaptive_simplex(prepared.layout, prepared.state, prepared.cell_marks(np.asarray((0,), dtype=np.int64)))
        if scenario == "coarsen_publication":
            single = D.coarsen_adaptive_simplex(
                prepared.layout, single.state, single.state.mesh.cell_active
            )
        reference = M.commit_adaptive_simplex(prepared, single.state)
        for result in results:
            target = result.target
            storage = target.mesh.storage
            assert storage is not None
            assert target.mesh.topology_id == reference.target.mesh.topology_id
            assert target.mesh.geometry_id == reference.target.mesh.geometry_id
            assert storage.global_entity_counts == ((4, 5, 2) if scenario == "coarsen_publication" else (5, 8, 4))
            assert target.collective_evidence.evidence_id == storage.evidence_id
            assert np.all(np.asarray(target.collective_evidence.global_checks))
            assert result.distribution is not None
            assert not result.distribution.rebalanced
            assert np.all(np.asarray(storage.local_neighborhood_complete))
            assert result.transfer.preserves_linear and result.transfer.preserves_constants
            assert not result.transfer.conservative
            original = 1 + source.mesh.coordinates[:, 0] + 2 * source.mesh.coordinates[:, 1]
            transferred = result.transfer.apply(original)
            np.testing.assert_allclose(transferred, 1 + target.mesh.coordinates[:, 0] + 2 * target.mesh.coordinates[:, 1], atol=1e-12)
            owned_cells = np.asarray(storage.entity_global_ids[2])[np.asarray(storage.entity_owned[2])]
            np.testing.assert_array_equal(np.asarray(result.hierarchy.cell_global_ids), np.sort(owned_cells))
            artifact = (result.hierarchy, result.lineage)
            provenance = ArrayArtifactProvenance(
                "native-bisection", (source.result_id, result.target.result_id),
                (storage.evidence_id,), (source.coordinate_contract.spatial_id,),
            )
            structure_ids = {"topology": target.mesh.topology_id, "coverage": storage.evidence_id}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "owner-lineage.zip"
                written = write_typed_array_artifact(path, artifact, artifact_kind="native-bisection-lineage",
                                                     provenance=provenance, structure_ids=structure_ids)
                template = jax.tree_util.tree_map(jnp.zeros_like, artifact)
                restored, receipt = read_typed_array_artifact(path, template,
                    artifact_kind="native-bisection-lineage", provenance=provenance, structure_ids=structure_ids)
                assert receipt.array_digest == written.array_digest
                np.testing.assert_array_equal(restored[0].cell_global_ids, np.sort(owned_cells))
                np.testing.assert_array_equal(restored[0].record_child_ids, result.hierarchy.record_child_ids)
                np.testing.assert_array_equal(restored[1].entity_lineage(2).source_global_ids,
                                              result.lineage.entity_lineage(2).source_global_ids)
            try:
                target.mesh.require_dense("implicit serial consumer")
            except ValueError:
                pass
            else:
                raise AssertionError("An owner-local carrier silently became a dense mesh")

        snapshots = snapshot_addressable_arrays({"arrays": dict(results[0].target.mesh.storage.logical_arrays)})
        cell_path = jax.tree_util.keystr((jax.tree_util.DictKey("arrays"), jax.tree_util.DictKey("epoch/cell_ids")))
        cell_shards = [shard for shard in snapshots if shard.array_path == cell_path]
        assert len(cell_shards) == 2
        for shard in cell_shards:
            rank = shard.index[0][0]
            held = np.asarray(results[rank].hierarchy.cell_global_ids)
            assert np.all(np.isin(held, shard.payload)), (rank, held, shard.payload)

        def halos_for(degree: int) -> tuple[Any, ...]:
            meshes = tuple(result.target.mesh for result in results)
            capacity = max(mesh.storage.entity_global_ids[degree].size for mesh in meshes)
            return prepare_owner_local_halo_plans(
                meshes, degree, devices=tuple(jax.devices()),
                axis_name="parts", message_capacity=capacity,
            )

        vertex_halos = halos_for(0)
        programs = tuple(prepare_owner_local_finite_element(
            FiniteElementPlan(result.target.mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))),
            halo, axis_name="parts",
        ) for result, halo in zip(results, vertex_halos, strict=True))
        def execution_fields(program: Any) -> Any:
            return (program.mass, program.stiffness, program.pairing, program.halo,
                    program.cell_owned, program.collective_metadata, program.global_ids,
                    program.owned_mask, program.local_discretization.mesh.coordinates)
        numeric, static = eqx.partition(execution_fields(programs[0]), eqx.is_array)
        _, structure = jax.tree_util.tree_flatten(numeric)
        leaves = [jax.tree_util.tree_leaves(eqx.filter(execution_fields(program), eqx.is_array)) for program in programs]
        packed = tuple(jnp.stack(values) for values in zip(*leaves, strict=True))
        def consume(local_leaves: Any) -> Any:
            fields = eqx.combine(jax.tree_util.tree_unflatten(structure, local_leaves), static)
            program = eqx.tree_at(execution_fields, programs[0], fields)
            coordinates = program.local_discretization.mesh.coordinates
            affine = 1 + coordinates[:, 0] + 2 * coordinates[:, 1]
            owned = jnp.where(program.owned_mask, affine, 0)
            mass = program.operator("mass")
            solved = program.solve_mass(mass.mv(owned), DistributedKrylovPolicy(16, relative_tolerance=1e-10))
            return (program.operator("stiffness").mv(jnp.where(program.owned_mask, 1.0, 0.0)), solved.solve.value, solved.accepted)
        constant, values, accepted = jax.pmap(consume, axis_name="parts", devices=jax.devices())(packed)
        np.testing.assert_allclose(constant, 0, atol=1e-12)
        np.testing.assert_array_equal(accepted, True)
        for rank, program in enumerate(programs):
            coordinates = program.local_discretization.mesh.coordinates
            np.testing.assert_allclose(values[rank], jnp.where(program.owned_mask, 1 + coordinates[:, 0] + 2 * coordinates[:, 1], 0), atol=1e-9)

        system = phx.equations.EulerSystem(2)
        cell_halos = halos_for(2)
        dynamics = []
        for result in results:
            storage = result.target.mesh.storage
            geometry = D.UnstructuredFiniteVolumePlan.from_cell_mesh(
                result.target.mesh, component_names=system.component_names,
                neighborhood_complete=storage.local_neighborhood_complete,
                physical_boundary_faces=storage.local_physical_boundary_facets,
                closure_evidence_id=storage.evidence_id,
            ).prepare(owned_face_capacity=8)
            boundaries = D.UnstructuredFiniteVolumeBoundarySet(
                geometry.boundary_patch_names,
                {name: D.SlipWallBoundary() for name in geometry.boundary_patch_names},
            )
            method = D.UnstructuredFiniteVolumeMethodPlan(
                D.PiecewiseConstantReconstruction(), D.RusanovFluxPlan()
            )
            dynamics.append(PreparedUnstructuredFiniteVolumeDynamics(system, geometry, method, boundaries))
        initial = []
        for program in dynamics:
            centroids = program.discretization.cell_centers
            primitive = jnp.stack(
                (1 + 0.25 * centroids[:, 0], jnp.zeros(centroids.shape[0]),
                 jnp.zeros(centroids.shape[0]), jnp.ones(centroids.shape[0])), axis=1,
            )
            initial.append(system.primitive_to_conserved(primitive))
        seeds = jnp.stack(tuple(
            jnp.where(program.discretization.cell_owned[:, None], value, 2 * value)
            for program, value in zip(dynamics, initial, strict=True)
        ))
        devices = tuple(jax.devices())
        group = ExecutionGroup(
            ExecutionGroupSpec(
                "accepted-owner-local-fv",
                tuple(sorted({device.process_index for device in devices})),
                tuple((device.process_index, device.id) for device in devices),
                mesh_axes=(("parts", len(devices)),),
            ),
            devices,
        )
        advance = execute_owner_local_finite_volume_advance(
            tuple(zip(dynamics, cell_halos, strict=True)),
            seeds, jnp.asarray(0.0), jnp.asarray(0.01), group, axis_name="parts",
        )
        updated = advance.state
        for rank, program in enumerate(dynamics):
            owned = program.discretization.cell_owned
            np.testing.assert_array_equal(updated[rank][~owned], seeds[rank][~owned])
        np.testing.assert_allclose(advance.inventory_after[0], advance.inventory_before[0], atol=1e-12)
        assert bool(advance.finite) and bool(advance.admissible)
        assert np.any(np.asarray(updated[0][dynamics[0].discretization.cell_owned, 0])
                      != np.asarray(initial[0][dynamics[0].discretization.cell_owned, 0]))

        L = phx.lifecycle
        result = results[0]
        topology = L.CompositionEntry(source, entry_id="mesh", role="topology", owner_id="meshing", structure_id=source.mesh.topology_id, revision_id=source.result_id, semantics_id="unit-square")
        old_values = 1 + source.mesh.coordinates[:, 0] + 2 * source.mesh.coordinates[:, 1]
        field = L.CompositionEntry(old_values, entry_id="u", role="physical-state", owner_id="fe", structure_id=source.mesh.topology_id, revision_id="source-field", semantics_id="affine-u", dependencies=(topology.binding("structure"),))
        composition = L.Composition((topology, field), boundary_id="accepted-step")
        successor = L.CompositionEntry(result.target, entry_id="mesh", role="topology", owner_id="meshing", structure_id=result.target.mesh.topology_id, revision_id=result.target.result_id, semantics_id="unit-square")
        target_values = result.transfer.apply(old_values)
        target_field = L.CompositionEntry(target_values, entry_id="u", role="physical-state", owner_id="fe", structure_id=result.target.mesh.topology_id, revision_id=result.transfer.transfer_id, semantics_id="affine-u", dependencies=(successor.binding("structure"),))
        transport = L.CompositionTransport("physical-remap", ("u",), (target_field,), source_structure_ids=(field.structure_id,), route_id=result.transfer.transfer_id, successful=jnp.all(result.target.collective_evidence.global_checks))
        rebind = L.CompositionRebind(composition, reprepare=(successor,), transports=(transport,))
        rejected = L.commit_composition_rebind(rebind, accepted_boundary=False)
        assert not rejected.published and rejected.composition.composition_id == composition.composition_id
        receipt = L.commit_composition_rebind(rebind, accepted_boundary=True)
        assert receipt.published
        np.testing.assert_array_equal(receipt.composition.entry("u").value, target_values)
    """
)


# MIGRATION: semantic IDs, numerical content, resources and atomic refusal.
@pytest.mark.parametrize("scenario", ("positive", "overflow", "shared_field_conflict"))
def test_native_owner_local_semantic_migration(scenario: str) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    subprocess.run(
        [sys.executable, "-c", _MIGRATION_SCRIPT, scenario],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )


_MIGRATION_SCRIPT = textwrap.dedent(
    """
    import sys
    import jax
    import jax.numpy as jnp
    import numpy as np
    from phydrax.discretization._adaptive_simplex import AdaptiveSimplexParts, AdaptiveSimplexStatus
    from phydrax.meshing._distribution import SimplexNeighborhoodWorkset
    from phydrax.meshing._distribution_migration import (
        OwnerLocalMigrationState, prepare_ownerlocal_migration, accept_ownerlocal_migration,
    )

    scenario = sys.argv[1]
    capacity = 1 if scenario == "overflow" else 2
    scientific = 2**54
    ids = np.full((2, capacity), np.iinfo(np.int64).max, dtype=np.int64)
    ids[:, 0] = [scientific+11, scientific+29]
    vertices = np.zeros((2, capacity, 3), dtype=np.int64)
    vertices[:, 0] = [[scientific+1, scientific+2, scientific+3], [scientific+2, scientific+4, scientific+3]]
    coordinates = np.zeros((2, capacity, 3, 2), dtype=np.float64)
    coordinates[:, 0] = [[[0.,0.], [1.,0.], [0.,1.]], [[1.,0.], [1.,1.], [0.,1.]]]
    valid = np.zeros((2, capacity), dtype=np.bool_)
    valid[:,0] = True
    owner = np.broadcast_to(np.arange(2,dtype=np.int32)[:,None], (2,capacity)).copy()
    exterior = np.zeros((2,capacity,3), dtype=np.bool_)
    exterior[:,0] = [[False,True,True], [True,False,True]]
    ancestry = np.full((2,capacity,3,3), -1, dtype=np.int64)
    weights = np.zeros((2,capacity,3,3), dtype=np.float64)
    for p in range(2):
        for v in range(3):
            ancestry[p,0,v,v] = vertices[p,0,v]
            weights[p,0,v,v] = 1.
    work = SimplexNeighborhoodWorkset(*map(jnp.asarray, (
        ids, vertices, coordinates, owner, np.zeros((2,capacity),np.int32),
        exterior, valid, ids.copy(), ids[...,None].copy(), ancestry, weights, np.zeros(2,np.int32),
    )))
    fields = np.zeros((2,capacity), np.float64)
    fields[:,0] = [3.,7.]
    vertex_fields = coordinates.sum(axis=-1)
    if scenario == "shared_field_conflict":
        vertex_fields[1,0,0] += 1.
    epochs = np.zeros((2,capacity),np.int64)
    epochs[:,0] = [41,42]
    vertex_owner = np.zeros((2,capacity,3),np.int32)
    vertex_owner[1,0] = [0,1,0]
    original = OwnerLocalMigrationState(
        work, jnp.asarray(fields), jnp.asarray(vertex_fields),
        jnp.asarray(epochs), jnp.asarray(vertex_owner),
    )
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=((0,1),(1,0)))
    targets = np.zeros((2,capacity),np.int32)
    if scenario != "overflow":
        targets[0,0] = 1
    prepared = prepare_ownerlocal_migration(
        parts, original, jnp.asarray(targets),
        halo_width=0 if scenario == "overflow" else 1, vertex_capacity=4,
    )
    target = accept_ownerlocal_migration(prepared)
    if scenario == "positive":
        np.testing.assert_array_equal(prepared.evidence.accepted, [True,True])
        np.testing.assert_array_equal(target.workset.cell_ids, [[scientific+11,scientific+29]]*2)
        np.testing.assert_array_equal(target.workset.cell_owner, [[1,0]]*2)
        np.testing.assert_array_equal(target.cell_fields, [[3.,7.]]*2)
        np.testing.assert_array_equal(target.cell_epochs, [[41,42]]*2)
        np.testing.assert_array_equal(target.workset.cell_vertices, [vertices[:,0]]*2)
        np.testing.assert_array_equal(target.workset.cell_coordinates, [coordinates[:,0]]*2)
        np.testing.assert_array_equal(target.workset.cell_exterior, [exterior[:,0]]*2)
        np.testing.assert_array_equal(target.workset.source_vertex_ids, [ancestry[:,0]]*2)
        np.testing.assert_array_equal(target.workset.source_weights, [weights[:,0]]*2)
        np.testing.assert_array_equal(target.vertex_fields, [vertex_fields[:,0]]*2)
        np.testing.assert_array_equal(target.workset.source_cell_ids, [[[scientific+11],[scientific+29]]]*2)
        np.testing.assert_array_equal(target.vertex_owners[0], [[1,0,0],[0,0,0]])
        owned = np.asarray(target.workset.cell_owner) == np.arange(2)[:,None]
        assert np.sum(np.asarray(target.cell_fields)[owned]) == 10.
        np.testing.assert_array_equal(prepared.evidence.sent_cells, [1,1])
        np.testing.assert_array_equal(prepared.evidence.received_cells, [1,1])
        np.testing.assert_array_equal(prepared.evidence.vertices_after, [4,4])
        assert prepared.evidence.route_metadata_bytes == 4
        assert prepared.target_parts.neighbor_pairs == ((0,1),(1,0))
    else:
        np.testing.assert_array_equal(prepared.evidence.accepted, [False,False])
        flag = AdaptiveSimplexStatus.CAPACITY_EXCEEDED if scenario == "overflow" else AdaptiveSimplexStatus.INVALID_GEOMETRY
        assert np.all(np.asarray(prepared.evidence.status) & int(flag))
        for old,new in zip(jax.tree_util.tree_leaves(original),jax.tree_util.tree_leaves(target),strict=True):
            np.testing.assert_array_equal(old,new)
    """
)


_COARSEN_SCRIPT = r"""
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexLayout, AdaptiveSimplexParts, AdaptiveSimplexStatus,
    adaptive_simplex_state, refine_adaptive_simplex_parts, coarsen_adaptive_simplex_parts,
)
count = int(sys.argv[1])
points = np.asarray([[0.,0.,-1.], [1.,0.,0.], [0.,1.,0.], [-1.,0.,0.], [0.,-1.,0.], [0.,0.,1.]])
rows = np.asarray([[0,1,2,5], [0,2,3,5], [0,3,4,5], [0,4,1,5]], dtype=np.int32)
layout = AdaptiveSimplexLayout(3, 3, vertex_capacity=64, cell_capacity=64, protected_edge_capacity=1, maximum_closure_iterations=64, maximum_coarsening_passes=1)
def piece(indices):
    ids = np.unique(rows[indices]).astype(np.int64)
    local = np.searchsorted(ids, rows[indices]).astype(np.int32)
    n = len(indices)
    return adaptive_simplex_state(layout, coordinates=points[ids], vertex_ids=ids,
        vertex_active=np.ones(len(ids), dtype=np.bool_), vertex_parents=np.full((len(ids),2),-1,dtype=np.int32),
        vertex_protected=np.zeros(len(ids),dtype=np.bool_), cells=local, tuples=local,
        tags=np.full(n,3,dtype=np.int32), blocks=np.zeros(n,dtype=np.int32), generations=np.zeros(n,dtype=np.int32),
        parents=np.full(n,-1,dtype=np.int32), children=np.full((n,2),-1,dtype=np.int32),
        bisection_vertices=np.full(n,-1,dtype=np.int32), cell_ids=indices.astype(np.int64)+10,
        cell_active=np.ones(n,dtype=np.bool_), cell_classes=np.full(n,7,dtype=np.int32),
        facet_classes=np.zeros((n,4),dtype=np.int32), protected_edges=np.empty((0,2),dtype=np.int32),next_vertex_id=6,next_cell_id=14)
pieces = [piece(np.arange(p*4//count,(p+1)*4//count,dtype=np.int32)) for p in range(count)]
source = jax.tree_util.tree_map(lambda *v:jnp.stack(v),*pieces)
parts = AdaptiveSimplexParts(jax.devices('cpu')[:count], neighbor_pairs=tuple((s,t) for s in range(count) for t in range(count) if s!=t))
refined = refine_adaptive_simplex_parts(layout, parts, source, source.mesh.cell_active)
assert np.all(np.asarray(refined.report.status)==0), refined.report.status
split = refined.state
# Admission and unique-vertex counting must propagate beyond a direct neighbor.
parts = AdaptiveSimplexParts(
    jax.devices("cpu")[:count],
    neighbor_pairs=tuple(
        route for p in range(count - 1) for route in ((p, p + 1), (p + 1, p))
    ),
)
full = coarsen_adaptive_simplex_parts(layout, parts, split, split.mesh.cell_active)
assert np.all(np.asarray(full.report.status)==0),full.report.status
assert np.all(np.asarray(full.report.operations)==4)
assert np.all(np.asarray(full.report.vertices)==1)
assert np.all(np.asarray(full.report.accepted)==8)
np.testing.assert_array_equal(full.state.mesh.cell_active,source.mesh.cell_active)
np.testing.assert_array_equal(full.state.mesh.vertex_active,source.mesh.vertex_active)
for p in range(count):
    active = np.asarray(full.state.mesh.cell_active[p])
    np.testing.assert_array_equal(full.state.mesh.cell_ids[p][active],source.mesh.cell_ids[p][active])
    np.testing.assert_array_equal(full.state.mesh.cells[p][active],source.mesh.cells[p][active])
    np.testing.assert_array_equal(full.state.mesh.coordinates[p][np.asarray(full.state.mesh.vertex_active[p])],source.mesh.coordinates[p][np.asarray(source.mesh.vertex_active[p])])
np.testing.assert_array_equal(full.state.cursors,split.cursors)
np.testing.assert_array_equal(full.state.retired,split.mesh.cell_active)
mid = int(np.flatnonzero(np.asarray(split.mesh.vertex_ids[-1])==6)[0])
leaf = int(np.flatnonzero(np.asarray(split.mesh.cell_active[-1]))[0])
for scenario in ('partial','class','facet','protected'):
    state=split
    marks=split.mesh.cell_active
    if scenario=='partial': marks=marks.at[-1,leaf].set(False)
    if scenario=='class': state=eqx.tree_at(lambda s:s.cell_classes,state,state.cell_classes.at[-1,leaf].set(9))
    if scenario=='facet':
        # A half of a facet split by the shared midpoint must keep its class.
        columns=np.flatnonzero(np.asarray(state.mesh.cells[-1,leaf])!=mid)
        # Facets opposite an unbisected equatorial vertex are the split halves.
        equatorial=next(int(c) for c in columns if int(state.mesh.vertex_ids[-1,state.mesh.cells[-1,leaf,c]]) not in (0,5))
        state=eqx.tree_at(lambda s:s.facet_classes,state,state.facet_classes.at[-1,leaf,equatorial].set(11))
    if scenario=='protected': state=eqx.tree_at(lambda s:s.vertex_protected,state,state.vertex_protected.at[-1,mid].set(True))
    update=coarsen_adaptive_simplex_parts(layout,parts,state,marks)
    assert np.all(np.asarray(update.report.status)==0),(scenario,update.report.status)
    assert np.all(np.asarray(update.report.operations)==0),scenario
    assert np.all(np.asarray(update.report.accepted)==0),scenario
    np.testing.assert_array_equal(update.state.mesh.cell_active,state.mesh.cell_active)
    np.testing.assert_array_equal(update.state.mesh.vertex_active,state.mesh.vertex_active)
    np.testing.assert_array_equal(update.state.retired,state.retired)
    np.testing.assert_array_equal(update.report.rejected,marks)
# Corrupt only an inactive source parent: restoration must fail atomically.
bad=eqx.tree_at(lambda s:s.mesh.cells,split,split.mesh.cells.at[-1,0].set(split.mesh.cells[-1,0,jnp.asarray([1,0,2,3],dtype=jnp.int32)]))
failed=coarsen_adaptive_simplex_parts(layout,parts,bad,bad.mesh.cell_active)
assert np.all(np.asarray(failed.report.status)&int(AdaptiveSimplexStatus.INVALID_GEOMETRY))
assert np.all(np.asarray(failed.report.operations)==0)
np.testing.assert_array_equal(failed.report.rejected,bad.mesh.cell_active)
cleared=eqx.tree_at(lambda s:s.clocks,failed.state,failed.state.clocks.at[:,2].set(0))
for left,right in zip(jax.tree_util.tree_leaves(cleared),jax.tree_util.tree_leaves(bad),strict=True):np.testing.assert_array_equal(left,right)
refused=coarsen_adaptive_simplex_parts(layout,parts,failed.state,failed.state.mesh.cell_active)
assert np.all(np.asarray(refused.report.failed))
assert np.all(np.asarray(refused.report.iterations)==0)
for left,right in zip(jax.tree_util.tree_leaves(refused.state),jax.tree_util.tree_leaves(failed.state),strict=True):np.testing.assert_array_equal(left,right)
# Separate component forests exercise a genuine unfinished lawful round.
offset=jnp.arange(count,dtype=jnp.int64)[:,None]*100
independent=eqx.tree_at(lambda s:(s.mesh.vertex_ids,s.mesh.cell_ids,s.cursors),source,
    (jnp.where(source.mesh.vertex_ids>=0,source.mesh.vertex_ids+offset,-1),
     jnp.where(source.mesh.cell_ids>=0,source.mesh.cell_ids+offset,-1),
     source.cursors.at[:,2].set(count*100+6).at[:,3].set(count*100+14)))
isolated=AdaptiveSimplexParts(jax.devices('cpu')[:count],neighbor_pairs=())
first=refine_adaptive_simplex_parts(layout,isolated,independent,independent.mesh.cell_active)
deep=refine_adaptive_simplex_parts(layout,isolated,first.state,first.state.mesh.cell_active)
assert np.all(np.asarray(deep.report.status)==0),deep.report.status
limited=coarsen_adaptive_simplex_parts(layout,isolated,deep.state,deep.state.mesh.cell_active)
assert np.all(np.asarray(limited.report.status)==int(AdaptiveSimplexStatus.PASS_LIMIT)),limited.report.status
assert np.all(np.asarray(limited.report.iterations)==1)
finished=limited
for _ in range(8):
    finished=coarsen_adaptive_simplex_parts(layout,isolated,finished.state,finished.state.mesh.cell_active)
    if np.all(np.asarray(finished.report.status)==0): break
    assert np.all(np.asarray(finished.report.status)==int(AdaptiveSimplexStatus.PASS_LIMIT))
else: raise AssertionError("bounded isolated forest did not finish")
np.testing.assert_array_equal(finished.state.mesh.cell_active,independent.mesh.cell_active)
np.testing.assert_array_equal(finished.state.mesh.vertex_active,independent.mesh.vertex_active)
print(f'COARSEN {count} devices: source IDs/arrays restored; remote vetoes; unique midpoint; retired slots; rollback/terminal; truthful pass limit')
"""


@pytest.mark.parametrize("device_count", (2, 4))
def test_collective_complete_family_coarsening(device_count: int) -> None:
    """Shared midpoint vetoes, identity restoration and collective rollback."""

    environment = dict(os.environ)
    environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    subprocess.run(
        [sys.executable, "-c", _COARSEN_SCRIPT, str(device_count)],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )


_PROOF_SCRIPT = r"""
import sys
from fractions import Fraction
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexLayout, AdaptiveSimplexParts, adaptive_simplex_state,
    refine_adaptive_simplex_parts, coarsen_adaptive_simplex_parts,
)
from phydrax.meshing._distribution import (
    _compiled_bisection_checks, expand_partitioned_simplex_neighborhood,
)
points = np.asarray([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
rounded = len(sys.argv)>1 and sys.argv[1]=="rounded"
if rounded:
    points=np.asarray([[.1,.2],[.4,.2],[.3,.7],[0.,.7]])
rows = np.asarray([[0,1,2],[2,3,0]],np.int32)
layout = AdaptiveSimplexLayout(2,2,vertex_capacity=32,cell_capacity=32,protected_edge_capacity=1,
    maximum_closure_iterations=64,maximum_coarsening_passes=64)
def piece(rank):
    ids=np.unique(rows[rank]).astype(np.int64)
    cells=np.searchsorted(ids,rows[rank:rank+1]).astype(np.int32)
    return adaptive_simplex_state(layout,coordinates=points[ids],vertex_ids=ids,
        vertex_active=np.ones(3,np.bool_),vertex_parents=np.full((3,2),-1,np.int32),
        vertex_protected=np.zeros(3,np.bool_),cells=cells,tuples=cells,tags=np.full(1,2,np.int32),
        blocks=np.zeros(1,np.int32),generations=np.zeros(1,np.int32),parents=np.full(1,-1,np.int32),
        children=np.full((1,2),-1,np.int32),bisection_vertices=np.full(1,-1,np.int32),
        cell_ids=np.asarray([rank+10],np.int64),cell_active=np.ones(1,np.bool_),
        cell_classes=np.full(1,7,np.int32),facet_classes=np.asarray([[3,0,3]],np.int32),
        protected_edges=np.empty((0,2),np.int32),next_vertex_id=4,next_cell_id=12)
source=jax.tree_util.tree_map(lambda *v:jnp.stack(v),piece(0),piece(1))
parts=AdaptiveSimplexParts(jax.devices(),neighbor_pairs=((0,1),(1,0)))
exterior=jnp.zeros(source.mesh.cells.shape,np.bool_).at[:,0,:].set(jnp.asarray([True,False,True]))
first=refine_adaptive_simplex_parts(layout,parts,source,source.mesh.cell_active).state
assert np.all(np.asarray(_compiled_bisection_checks(parts,first,source,exterior)))
first_exterior=first.facet_classes!=0
assert np.any(np.asarray(first.parents)>=0) # accepted predecessor is not a root-only echo
second=refine_adaptive_simplex_parts(layout,parts,first,first.mesh.cell_active).state
assert np.all(np.asarray(_compiled_bisection_checks(parts,second,first,first_exterior)))
def packets(state,initial,flags):
    return expand_partitioned_simplex_neighborhood(parts,state,initial,flags,
        halo_width=1,cell_capacity=32,vertex_capacity=32)
work=packets(second,first,first_exterior)
assert np.all(np.asarray(work.status)==0),work.status
initial_ids=np.asarray(first.mesh.cell_ids)[np.asarray(first.mesh.cell_active)]
valid=np.asarray(work.cell_valid)
assert np.all(np.isin(np.asarray(work.root_cell_ids)[valid],initial_ids))
assert not np.any(np.isin(np.asarray(work.root_cell_ids)[valid],[10,11]))
# The genuine new midpoint's field support is on the immediate accepted
# predecessor vertices, including its issued shared midpoint, not original roots.
support=np.asarray(work.source_vertex_ids)[valid]
assert np.any(support==4)
field=np.arange(64,dtype=np.float64)**2
values=np.sum(np.where(support>=0,field[np.maximum(support,0)],0)*np.asarray(work.source_weights)[valid],axis=2)
vertices=np.asarray(work.cell_vertices)[valid]
expected={}
for rank in range(2):
    count=int(second.cursors[rank,0])
    identifiers=np.asarray(second.mesh.vertex_ids[rank])
    local=field[np.maximum(identifiers,0)].copy()
    for slot in range(int(first.cursors[rank,0]),count):
        a,b=np.asarray(second.vertex_parents[rank,slot])
        local[slot]=.5*local[a]+.5*local[b]
    expected.update((int(identifiers[slot]),local[slot]) for slot in range(count))
for vertex in np.unique(vertices):
    copies=values[vertices==vertex]
    np.testing.assert_allclose(copies,copies[0],rtol=0,atol=0)
    np.testing.assert_allclose(copies,expected[int(vertex)],rtol=0,atol=0)
coarse=coarsen_adaptive_simplex_parts(layout,parts,first,first.mesh.cell_active).state
assert np.all(np.asarray(_compiled_bisection_checks(parts,coarse,first,first_exterior)))
work=packets(coarse,first,first_exterior)
assert np.all(np.asarray(work.status)==0),work.status
for rank in range(2):
    for lane in np.flatnonzero(np.asarray(work.cell_valid[rank])):
        identifier=int(work.cell_ids[rank,lane])
        owner=identifier-10
        children=np.asarray(first.mesh.cell_ids[owner])[np.asarray(first.mesh.cell_active[owner])]
        actual=np.asarray(work.source_cell_ids[rank,lane])
        np.testing.assert_array_equal(np.sort(actual[actual>=0]),np.sort(children))
        assert int(work.root_cell_ids[rank,lane])==-1
        ids=np.asarray(work.source_vertex_ids[rank,lane])
        weights=np.asarray(work.source_weights[rank,lane])
        reconstructed=np.sum(np.where(ids>=0,field[np.maximum(ids,0)],0)*weights,axis=1)
        np.testing.assert_array_equal(reconstructed,field[np.asarray(work.cell_vertices[rank,lane])])
# An immutable ancestor corrupted on one owner prevents the collective verdict.
bad=eqx.tree_at(lambda s:s.parents,second,second.parents.at[1,1].set(-1))
checks=np.asarray(_compiled_bisection_checks(parts,bad,first,first_exterior))
assert not np.all(checks[:,0])
# Prior retired records remain bound on a later refinement epoch.
again=refine_adaptive_simplex_parts(layout,parts,coarse,coarse.mesh.cell_active).state
coarse_exterior=coarse.facet_classes!=0
assert np.all(np.asarray(_compiled_bisection_checks(parts,again,coarse,coarse_exterior)))
retired=int(np.flatnonzero(np.asarray(coarse.retired[1]))[0])
bad=eqx.tree_at(lambda s:s.mesh.cell_ids,again,again.mesh.cell_ids.at[1,retired].add(100))
assert not np.all(np.asarray(_compiled_bisection_checks(parts,bad,coarse,coarse_exterior))[:,0])
print("PROOF: accepted forest refine/coarsen; complete source children; nonlinear ghost field ancestry; owner corruption rejected")

# Scientific organization and real source coefficient maps survive the same
# numerical forest. These are occurrence arrays, not positive report flags.
import phydrax as phx
from jax.sharding import NamedSharding, PartitionSpec
from phydrax.meshing._device_adaptation import _canonical_rows
from phydrax.meshing._distribution import _facet_packets
from phydrax.meshing._collective_organization import build_collective_organization_witness
from phydrax.meshing._collective_geometry import build_collective_geometry_witness
from phydrax.meshing.providers._native_publication import NativeCertificationRequest, publish_native_result
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
M=phx.meshing
D=phx.discretization
mesh=D.CellMesh(points,(D.CellBlock("domain","triangle",rows,global_ids=np.asarray([10,11],np.int64)),),
    vertex_global_ids=np.arange(4,dtype=np.int64))
def scope(degree,ids):
    return M.MeshingScope(mesh.mesh_id,mesh.numeric_version,M.MeshingEntityKind.MESH,degree,
        mesh.entity_set(degree).entity_set_id,np.asarray(ids,np.int64))
all_cells=scope(2,[10,11])
zones=(M.MeshZone("material",M.MeshZoneRole.USER,all_cells),)
labels=(M.MeshLabel("left",scope(2,[10])),)
attributes=(M.MeshAttribute("marker",M.MeshAttributeRole.MARKER,all_cells,np.asarray([19,23],np.int64)),)
seed=M.certify_cell_mesh(mesh,phx.SpatialCoordinateContract.si())
loop=np.asarray([[0,1],[1,2],[2,3],[3,0]],np.int32)
domain=phx.geometry.PiecewiseLinearDomain(points,loop,np.tile(np.asarray([0,-1],np.int32),(4,1)),("domain",),source_id="proof-source")
original=publish_native_result(mesh,seed.coordinate_contract,M.MeshingComplianceReport("proof-source"),(),
    seed.provider,{"kind":"scientific-proof-source"},
    NativeCertificationRequest(M.MeshCertificationSchedule("volume_plc"),"proof-source","r0",M.MeshingLimits(),
        domain=domain,cell_regions=np.zeros(2,np.int32)),
    audit_policy=M.CellMeshAuditPolicy(watertight_boundary=M.CellMeshAuditDisposition.REJECT),
    derivative_mode=M.MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,enforced_limits=(),unenforced_limits=(),
    zones=zones,labels=labels,attributes=attributes)
placement=NamedSharding(parts.mesh,PartitionSpec(parts.axis_name))
def logical(state):
    vkeys,vorder,nv=_canonical_rows(state.mesh.vertex_ids.reshape((-1,1)),state.mesh.vertex_active.reshape(-1))
    ckeys,corder,nc=_canonical_rows(state.mesh.cell_ids.reshape((-1,1)),state.mesh.cell_active.reshape(-1))
    fkeys,_=jax.vmap(_facet_packets)(state)
    fkeys,forder,nf=_canonical_rows(fkeys.reshape((-1,2)),jnp.repeat(state.mesh.cell_active.reshape(-1),3))
    ids=(vkeys[:,0],jnp.where(jnp.arange(fkeys.shape[0])<nf,jnp.arange(fkeys.shape[0],dtype=jnp.int64)+1000,-1),ckeys[:,0])
    owners=(vorder//32,forder//(32*3),corder//32)
    return tuple(jax.device_put(a,placement) for a in (vkeys,fkeys,ckeys)),tuple(jax.device_put(a,placement) for a in ids),tuple(jax.device_put(a.astype(jnp.int32),placement) for a in owners),(nv,nf,nc)
for state in (first,second,coarse):
    keys,ids,owners,counts=logical(state)
    bank,organization_id=build_collective_organization_witness(original,state,keys,ids,counts,uniform_refinement=None)
    bank=dict(bank)
    roots=np.asarray(bank["organization/cell/source_ids"])[:counts[-1]]
    np.testing.assert_array_equal(np.asarray(bank["organization/label/0/membership"])[:counts[-1]],roots==10)
    np.testing.assert_array_equal(np.asarray(bank["organization/attribute/0/values"])[:counts[-1]],np.where(roots==10,19,23))
    from phydrax._meshcore import NativeExecutionBudget
    with NativeExecutionBudget(max_work=40_960_000,max_geometry_queries=200_000,max_scratch_bytes=81_920_000,max_wall_seconds=120,max_cavity_cells=M.MeshingLimits().maximum_cavity_cells) as native:
        with native.host_workspace():
            geometry,geometry_id=build_collective_geometry_witness(original,state,tuple(bank.items()),ids,owners,counts)
    geometry=dict(geometry)
    np.testing.assert_array_equal(np.asarray(geometry["geometry/coordinates"])[:4],np.asarray(original.geometry.coordinates))
    if state is coarse:
        assert geometry_id==cell_geometry_id(original.geometry)
        np.testing.assert_array_equal(np.asarray(geometry["geometry/action_counts/domain"])[:2],np.zeros(2,dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(geometry["geometry/action_weights/domain"])[:2],0.)
    else:
        assert geometry_id!=cell_geometry_id(original.geometry)
if rounded:
    # The retained scientific midpoint is exact over the original FP64 field,
    # even though the topological FP64 carrier cannot represent that value.
    keys,ids,owners,counts=logical(first)
    bank,_=build_collective_organization_witness(original,first,keys,ids,counts,uniform_refinement=None)
    bank=dict(bank)
    row=int(np.flatnonzero(np.asarray(ids[0])==4)[0])
    support=np.asarray(bank["organization/vertex/source_ids"])[row]
    weights=np.asarray(bank["organization/vertex/source_weights"])[row]
    exact=sum((Fraction.from_float(float(points[i,0]))*Fraction.from_float(float(w))
        for i,w in zip(support,weights,strict=True) if i>=0),Fraction())
    actual=first.mesh.coordinates[0,int(np.flatnonzero(np.asarray(first.mesh.vertex_ids[0])==4)[0]),0]
    assert exact!=Fraction.from_float(float(actual))
    assert exact==(Fraction.from_float(float(points[0,0]))+Fraction.from_float(float(points[2,0])))/2
print("PROOF: global source occurrences and exact original coefficient maps; coarsen restores scientific field")
"""


@pytest.mark.parametrize("coordinate_kind", ("dyadic", "rounded"))
def test_accepted_predecessor_forest_and_scientific_neighborhood_ancestry(
    coordinate_kind: str,
) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", _PROOF_SCRIPT, coordinate_kind],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + "\n" + completed.stderr, pytrace=False)


# FOREST MIGRATION: retired/removal records and typed history remain scientific state.
@pytest.mark.parametrize("scenario", ("positive", "forest_overflow"))
def test_complete_forest_migration_and_cross_owner_coarsening(scenario: str) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", _FOREST_MIGRATION_SCRIPT, scenario],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode:
        pytest.fail(f"{completed.stdout}\n{completed.stderr}", pytrace=False)


_FOREST_MIGRATION_SCRIPT = r"""
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexLayout, AdaptiveSimplexParts, AdaptiveSimplexStatus,
    adaptive_simplex_state, refine_adaptive_simplex_parts, coarsen_adaptive_simplex_parts,
)
from phydrax.meshing._distribution import SimplexNeighborhoodWorkset
from phydrax.meshing._distribution_migration import (
    OwnerLocalMigrationState, prepare_ownerlocal_migration, accept_ownerlocal_migration,
    ownerlocal_forest_marks, ownerlocal_solver_cell_owners, ownerlocal_solver_forest,
    ownerlocal_simplex_graph, restart_solver_target_owners,
)
scenario = sys.argv[1]
capacity = 16 if scenario == "forest_overflow" else 32
points = np.asarray([[0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
rows = np.asarray([[0,1,2,5],[0,2,3,5],[0,3,4,5],[0,4,1,5]], np.int32)
layout = AdaptiveSimplexLayout(3, 3, vertex_capacity=32, cell_capacity=capacity,
    protected_edge_capacity=4, maximum_closure_iterations=32, maximum_coarsening_passes=2)
def piece(indices):
    ids = np.unique(rows[indices]).astype(np.int64)
    local = np.searchsorted(ids, rows[indices]).astype(np.int32)
    n = len(indices)
    return adaptive_simplex_state(layout, coordinates=points[ids], vertex_ids=ids,
        vertex_active=np.ones(len(ids), np.bool_), vertex_parents=np.full((len(ids),2),-1,np.int32),
        vertex_protected=np.zeros(len(ids), np.bool_), cells=local, tuples=local,
        tags=np.full(n,3,np.int32), blocks=np.zeros(n,np.int32), generations=np.zeros(n,np.int32),
        parents=np.full(n,-1,np.int32), children=np.full((n,2),-1,np.int32),
        bisection_vertices=np.full(n,-1,np.int32), cell_ids=indices.astype(np.int64)+10,
        cell_active=np.ones(n,np.bool_), cell_classes=np.full(n,7,np.int32),
        facet_classes=np.zeros((n,4),np.int32), protected_edges=np.empty((0,2),np.int32),
        next_vertex_id=6, next_cell_id=14)
source = jax.tree_util.tree_map(lambda *a:jnp.stack(a),
    piece(np.arange(2,dtype=np.int32)), piece(np.arange(2,4,dtype=np.int32)))
parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=((0,1),(1,0)))
first = refine_adaptive_simplex_parts(layout, parts, source, source.mesh.cell_active)
assert np.all(np.asarray(first.report.status)==0)
undo = coarsen_adaptive_simplex_parts(layout, parts, first.state, first.state.mesh.cell_active)
assert np.all(np.asarray(undo.report.status)==0)
second = refine_adaptive_simplex_parts(layout, parts, undo.state, undo.state.mesh.cell_active)
assert np.all(np.asarray(second.report.status)==0)
split = second.state
assert np.any(np.asarray(split.vertex_removal)>=0)
split = eqx.tree_at(
    lambda state:(state.vertex_protected,state.protected_codes), split,
    (split.vertex_removal>=0,
     jnp.broadcast_to(jnp.asarray([1,*([np.iinfo(np.int64).max]*3)],jnp.int64),(2,4))),
)
mesh = split.mesh
P,C = mesh.cell_ids.shape
V = mesh.vertex_ids.shape[1]
W = 4
vids = jnp.take_along_axis(mesh.vertex_ids[:,:,None], mesh.cells, axis=1)
coords = jax.vmap(lambda x,r:x[r])(mesh.coordinates,mesh.cells)
rank = jnp.arange(P,dtype=jnp.int32)[:,None]
co = jnp.where(mesh.cell_ids>=0,rank,-1)
vo = np.full((P,V),-1,np.int32)
for p in range(P):
    for v,identifier in enumerate(np.asarray(mesh.vertex_ids[p])):
        if identifier>=0:
            vo[p,v] = min(q for q in range(P) if identifier in np.asarray(mesh.vertex_ids[q]))
work = SimplexNeighborhoodWorkset(
    jnp.where(mesh.cell_active,mesh.cell_ids,jnp.iinfo(jnp.int64).max), vids, coords,
    jnp.broadcast_to(rank,(P,C)), split.cell_classes, mesh.facet_neighbors<0, mesh.cell_active,
    mesh.cell_ids, mesh.cell_ids[:,:,None], jnp.full((P,C,W,W),-1,jnp.int64),
    jnp.zeros((P,C,W,W),jnp.float64), jnp.zeros(P,jnp.int32))
history = (mesh.cell_ids+2**54, split.retired,
           mesh.cell_ids[:,:,None]*jnp.asarray([2.,3.],jnp.float64))
vhistory = (mesh.vertex_ids+2**54, split.vertex_removal, mesh.coordinates)
original = OwnerLocalMigrationState(work,
    jnp.where(mesh.cell_active,mesh.cell_ids.astype(jnp.float64),0), coords, mesh.cell_ids,
    jnp.zeros((P,C,W),jnp.int32), split, co, jnp.asarray(vo), history, vhistory)
targets = jnp.broadcast_to((jnp.arange(C,dtype=jnp.int32)%2)[None,:],(P,C))
prepared = prepare_ownerlocal_migration(parts, original, targets, halo_width=1, vertex_capacity=16)
accepted = accept_ownerlocal_migration(prepared)
if scenario == "forest_overflow":
    assert np.all(np.asarray(prepared.evidence.status) & int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED))
    np.testing.assert_array_equal(prepared.evidence.accepted,[False,False])
    for old,new in zip(jax.tree_util.tree_leaves(original),jax.tree_util.tree_leaves(accepted),strict=True):
        np.testing.assert_array_equal(old,new)
else:
    np.testing.assert_array_equal(prepared.evidence.accepted,[True,True])
    from phydrax.graph._partition import partition_graph, GraphPartitionPlan
    from jax.sharding import NamedSharding, PartitionSpec
    placed = jax.device_put(original,NamedSharding(parts.mesh,PartitionSpec(parts.axis_name)))
    graph = ownerlocal_simplex_graph(parts,placed,
        vertex_weights=jnp.where(placed.workset.cell_valid,1+placed.workset.cell_ids%3,0).astype(jnp.int64))
    graph_result = partition_graph(graph,GraphPartitionPlan(2,maximum_imbalance=1.5))
    assert graph_result.evidence.accepted
    raw_proposal,proposal_status = restart_solver_target_owners(parts,placed.forest,placed.workset,graph_result.parts)
    np.testing.assert_array_equal(proposal_status,[0,0])
    for p in range(P):
        active_rows = np.asarray(placed.workset.cell_valid[p])
        proposal_by_id = dict(zip(np.asarray(placed.workset.cell_ids[p])[active_rows],
                                 np.asarray(graph_result.parts[p])[active_rows],strict=True))
        active_raw = np.asarray(placed.forest.mesh.cell_active[p])
        expected_proposal = [proposal_by_id[identifier] for identifier in np.asarray(placed.forest.mesh.cell_ids[p])[active_raw]]
        np.testing.assert_array_equal(np.asarray(raw_proposal[p])[active_raw],expected_proposal)
    np.testing.assert_array_equal(np.asarray(raw_proposal)[~np.asarray(placed.forest.mesh.cell_active)],-1)
    forest = accepted.forest
    assert forest is not None
    np.testing.assert_array_equal(
        np.sort(np.asarray(forest.mesh.cell_ids)[np.asarray(forest.mesh.cell_ids)>=0]),
        np.sort(np.asarray(split.mesh.cell_ids)[np.asarray(split.mesh.cell_ids)>=0]))
    for p in range(P):
        ids = np.asarray(forest.mesh.cell_ids[p])
        active = ids>=0
        np.testing.assert_array_equal(np.asarray(accepted.cell_history[0][p])[active],ids[active]+2**54)
        np.testing.assert_array_equal(np.asarray(accepted.cell_history[1][p])[active],np.asarray(forest.retired[p])[active])
        np.testing.assert_array_equal(np.asarray(accepted.cell_history[2][p])[active],ids[active,None]*[2.,3.])
        vertices = np.asarray(forest.mesh.vertex_ids[p])
        active = vertices>=0
        np.testing.assert_array_equal(np.asarray(accepted.vertex_history[0][p])[active],vertices[active]+2**54)
        np.testing.assert_array_equal(np.asarray(accepted.vertex_history[1][p])[active],np.asarray(forest.vertex_removal[p])[active])
        np.testing.assert_array_equal(np.asarray(accepted.vertex_history[2][p])[active],np.asarray(forest.mesh.coordinates[p])[active])
    np.testing.assert_array_equal(forest.clocks,split.clocks)
    np.testing.assert_array_equal(forest.counters,split.counters)
    for p in range(P):
        codes = np.asarray(forest.protected_codes[p])
        codes = codes[codes!=np.iinfo(np.int64).max]
        if codes.size:
            endpoints = np.asarray(forest.mesh.vertex_ids[p])[np.stack((codes//V,codes%V),axis=1)]
            np.testing.assert_array_equal(endpoints,[[0,1]])
        removed = np.asarray(forest.vertex_removal[p])>=0
        assert np.all(np.asarray(forest.vertex_protected[p])[removed])
    np.testing.assert_array_equal(prepared.evidence.forest_cells_before,[10,10])
    np.testing.assert_array_equal(prepared.evidence.forest_cells_after,[20,0])
    solver_owners,solver_owner_status = ownerlocal_solver_cell_owners(prepared.target_parts,accepted)
    np.testing.assert_array_equal(solver_owner_status,[0,0])
    for p in range(P):
        active = np.asarray(forest.mesh.cell_active[p])
        scientific_ids = np.asarray(forest.mesh.cell_ids[p])[active]
        np.testing.assert_array_equal(np.asarray(solver_owners[p])[active],(scientific_ids-22)%2)
    solver_forest,solver_cell_history,solver_vertex_history,solver_forest_status = ownerlocal_solver_forest(
        prepared.target_parts,accepted,
    )
    np.testing.assert_array_equal(solver_forest_status,[0,0])
    for p in range(P):
        allocated = np.asarray(solver_forest.mesh.cell_ids[p])>=0
        np.testing.assert_array_equal(
            np.asarray(solver_cell_history[0][p])[allocated],
            np.asarray(solver_forest.mesh.cell_ids[p])[allocated]+2**54,
        )
        resident_ids = np.asarray(accepted.workset.cell_ids[p])[np.asarray(accepted.workset.cell_valid[p])]
        assert np.all(np.isin(resident_ids,np.asarray(solver_forest.mesh.cell_ids[p])))
    old_owned = np.asarray(work.cell_valid) & (np.asarray(work.cell_owner)==np.arange(P)[:,None])
    new_owned = np.asarray(accepted.workset.cell_valid) & (np.asarray(accepted.workset.cell_owner)==np.arange(P)[:,None])
    assert np.sum(np.asarray(original.cell_fields)[old_owned])==np.sum(np.asarray(accepted.cell_fields)[new_owned])
    marks,mark_status = ownerlocal_forest_marks(prepared.target_parts,accepted,accepted.workset.cell_valid)
    np.testing.assert_array_equal(mark_status,[0,0])
    np.testing.assert_array_equal(marks,forest.mesh.cell_active)
    assert prepared.forest_parts is not None
    coarsened = coarsen_adaptive_simplex_parts(layout, prepared.forest_parts, forest, marks)
    assert np.all(np.asarray(coarsened.report.status)==0)
    np.testing.assert_array_equal(
        np.sort(np.asarray(coarsened.state.mesh.cell_ids)[np.asarray(coarsened.state.mesh.cell_active)]),
        [10,11,12,13])
    np.testing.assert_array_equal(coarsened.report.operations,[4,4])
    print("Retired/removed records, int64/bool/vector history, conservation and cross-owner sibling coarsening preserved")
"""


# PLC TRANSFER: two real process owners recertify and transfer exact planar strata.
def test_two_owner_bisection_transfers_exact_planar_plc_associations() -> None:
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        PHYDRAX_CPU_COLLECTIVES="gloo",
        PYTHONPATH=str(Path(phx.__file__).resolve().parents[1]),
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _PLC_TRANSFER_SCRIPT, coordinator, str(index)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(2)
    ]
    outputs = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=300)
            if process.returncode:
                pytest.fail(stdout + "\n" + stderr, pytrace=False)
            outputs.append(stdout)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    owners = [
        json.loads(line.removeprefix("PLC_VERDICTS "))
        for output in outputs
        for line in output.splitlines()
        if line.startswith("PLC_VERDICTS ")
    ]
    assert [owner["partition"] for owner in owners] == [0, 1]
    for dimension in ("0", "1", "2"):
        first, second = (owner["strata"][dimension] for owner in owners)
        shared = first.keys() & second.keys()
        # The same global entity carries one exact source stratum on every owner.
        assert shared and all(first[key] == second[key] for key in shared)
        assert len(first.keys() | second.keys()) == owners[0]["counts"][int(dimension)]
    # Pieces of authored source edges lie on both owners with one verdict.
    first, second = (owner["strata"]["1"] for owner in owners)
    assert any(first[key][0] == 1 for key in first.keys() & second.keys())


_PLC_TRANSFER_SCRIPT = r"""
import json
import sys
import jax
jax.distributed.initialize(coordinator_address=sys.argv[1], num_processes=2,
    process_id=int(sys.argv[2]), local_device_ids=[0])
from fractions import Fraction
import numpy as np
import phydrax as phx
from phydrax.meshing._association import PlcAssociationTransfer
from phydrax.meshing.providers._native_planar import _source_arrays
M = phx.meshing
D = phx.discretization
# A plate with a hole; its coarse native mesh satisfies the longest-edge
# matching condition, and the Morton cut crosses both boundary loops.
region = phx.geometry.PlanarMeshRegion(
    np.asarray([[0, 0], [2, 0], [2, 2], [0, 2], [0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]],
               dtype=np.float64),
    [[0, 1, 2, 3], [4, 5, 6, 7]], feature_id="plate")
authored = M.NativePlanarSource(region, "r1")
scope = M.MeshingScope("plate", "r1", M.MeshingEntityKind.GEOMETRY, 2, "plate-region", np.asarray([0]))
source = M.NativeMeshingProvider(M.NativeMeshingOptions("planar_constrained_delaunay")).plan(
    authored, M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))), scope,
        planar_embedding=phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        size_controls=(M.UniformSizeControl(scope, 1.0, maximum_size=1.5, strength=M.SizeControlStrength.HARD),),
        size_compliance=M.SizeCompliancePolicy(relative_tolerance=0.5, target_statistics=("p50",)),
    ), coordinate_contract=phx.SpatialCoordinateContract.si()).execute()
assert source.certification is not None and source.certification.coverage is not None
domain = source.certification.request.domain
_, edges, _ = _source_arrays(authored)
transfer = PlcAssociationTransfer(
    domain, source.coordinate_contract, authored.source_revision, edge_vertices=edges,
    triangle_vertices=np.empty((0, 3), dtype=np.int64), triangle_facets=np.empty((0,), dtype=np.int64),
    facet_regions=domain.facet_regions)
partition_policy = M.MeshPartitionPolicy(M.MeshPartitionKind.MORTON, 2, halo_width=2)
policy = M.MeshAdaptationPolicy(
    M.MeshAdaptationRoute.DEVICE_BISECTION,
    device_policy=D.AdaptiveSimplexPolicy(vertex_capacity=128, cell_capacity=128),
    distribution=M.prepare_mesh_distribution(M.MeshPart("domain", source), policy=partition_policy),
    partition_policy=partition_policy, association_transfer=transfer,
    audit_policy=M.CellMeshAuditPolicy(require_complete_association=True,
                                       watertight_boundary=M.CellMeshAuditDisposition.REJECT))
partitioned = M.partition_adaptive_simplex(M.prepare_adaptive_simplex(source, policy=policy))
update = D.refine_adaptive_simplex_parts(
    partitioned.layout, partitioned.parts, partitioned.states, partitioned.states.mesh.cell_active)
assert np.all(np.asarray(update.report.status.addressable_shards[0].data) == 0)
(result,) = M.commit_partitioned_adaptive_simplex(partitioned, update.state)
assert result.status is M.MeshAdaptationStatus.COMPLETE
target = result.target
storage = target.mesh.storage
assert storage is not None and storage.partition_index == jax.process_index()
assert target.collective_certificates is not None
embedding, coverage = target.collective_certificates
embedding.binding.require(target.mesh, target.geometry)
coverage.binding.require(target.mesh, target.geometry)
assert embedding.status == "certified" and not embedding.findings
assert coverage.status == "certified" and not coverage.findings
assert coverage.embedding_certificate_id == embedding.certificate_id
assert coverage.domain_id == domain.domain_id
assert coverage.premise_certificate_ids == (source.certification.coverage.certificate_id, storage.evidence_id)
assert coverage.covered_source_facet_count == coverage.source_facet_count
# Independent oracle: the plate area minus the hole.
assert coverage.requested_region_measures == coverage.achieved_region_measures == (3.0,)
charged, required = coverage.source_expression_work_units, coverage.source_expression_required_work_units
assert charged is not None and required is not None and required >= charged
# Exact strata, checked against the authored tables with rational predicates.
points = [tuple(Fraction(float(value)) for value in row) for row in np.asarray(target.mesh.coordinates)]
authority = [tuple(Fraction(float(value)) for value in row) for row in transfer.source_vertices]
def on_segment(point, index):
    (ax, ay), (bx, by) = (authority[vertex] for vertex in transfer.edge_vertices[index])
    return ((bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax) == 0
            and min(ax, bx) <= point[0] <= max(ax, bx) and min(ay, by) <= point[1] <= max(ay, by))
strata = {}
for association in target.associations:
    dimension = next(d for d in range(3) if association.target_entity_set_id == target.mesh.entity_set(d).entity_set_id)
    association.validate_target(target.mesh.entity_set(dimension))
    ids = np.asarray(association.target_global_ids, dtype=np.int64)
    assert association.exact and association.complete
    np.testing.assert_array_equal(np.sort(ids), np.sort(np.asarray(target.mesh.entity_set(dimension).entity_ids)))
    kinds = np.asarray(association.source_dimensions, dtype=np.int64)
    indices = np.asarray(association.source_indices, dtype=np.int64)
    if dimension == 0:
        rows = np.searchsorted(np.asarray(target.mesh.vertex_global_ids), ids)
        for row, kind, index in zip(rows.tolist(), kinds.tolist(), indices.tolist(), strict=True):
            point = points[row]
            match kind:
                case 0:
                    assert point == authority[int(np.flatnonzero(transfer.vertex_indices == index)[0])]
                case 1:
                    assert on_segment(point, int(np.flatnonzero(transfer.edge_indices == index)[0]))
                case 2:
                    assert not any(on_segment(point, edge) for edge in range(transfer.edge_vertices.shape[0]))
                case _:
                    raise AssertionError(f"Unknown planar source stratum {kind}")
    strata[str(dimension)] = {str(identifier): [kind, index] for identifier, kind, index in
                              zip(ids.tolist(), kinds.tolist(), indices.tolist(), strict=True)}
print("PLC_VERDICTS", json.dumps({
    "partition": storage.partition_index, "counts": list(storage.global_entity_counts),
    "strata": strata}), flush=True)
"""


# RESTART/REPACK: two real process owners become a genuinely different device axis.
@pytest.fixture(scope="module")
def _saved_raw_restart(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("raw-forest-restart")
    (directory / "repository").mkdir()
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        PHYDRAX_CPU_COLLECTIVES="gloo",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _REPACK_SCRIPT,
                "save",
                str(directory),
                str(index),
                coordinator,
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(2)
    ]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=180)
            if process.returncode:
                pytest.fail(stdout + "\n" + stderr, pytrace=False)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    return directory


@pytest.mark.parametrize("target_count", (1, 3))
def test_saved_complete_forest_reassembles_on_changed_device_count(
    _saved_raw_restart: Path,
    target_count: int,
) -> None:
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={target_count}",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _REPACK_SCRIPT,
            "restore",
            str(_saved_raw_restart),
            str(target_count),
            "positive",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + "\n" + completed.stderr, pytrace=False)


@pytest.mark.parametrize(
    "fault",
    (
        "duplicate",
        "missing",
        "overflow",
        "shared_conflict",
        "lost_authority",
        "packet_overflow",
    ),
)
def test_changed_owner_count_rejects_malformed_complete_epoch(
    _saved_raw_restart: Path,
    fault: str,
) -> None:
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _REPACK_SCRIPT,
            "restore",
            str(_saved_raw_restart),
            "1",
            fault,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + "\n" + completed.stderr, pytrace=False)


_REPACK_SCRIPT = r"""
import sys
from pathlib import Path
import equinox as eqx
import jax
stage, directory = sys.argv[1:3]
if stage == "save":
    jax.distributed.initialize(coordinator_address=sys.argv[4], num_processes=2,
        process_id=int(sys.argv[3]), local_device_ids=[0])
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental import multihost_utils
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexLayout, AdaptiveSimplexParts, AdaptiveSimplexState, MaskedSimplexMesh,
    adaptive_simplex_state, refine_adaptive_simplex_parts, coarsen_adaptive_simplex_parts,
    AdaptiveSimplexStatus,
)
from phydrax._execution_runtime import ExecutionRuntime
from phydrax.meshing._restart_distribution import repack_simplex_restart, restart_forest_target_owners
from phydrax.lifecycle._distributed_checkpoint import (
    publish_process_checkpoint, assemble_distributed_checkpoint_from_repository,
    restore_global_array_from_checkpoint,
)
from phydrax.lifecycle._repository import HPCFilesystemProfile, POSIXArtifactRepository, POSIXRepositoryPolicy
root = Path(directory)
profile = HPCFilesystemProfile("posix.raw-restart","local-posix",
    atomic_rename_same_filesystem=True,file_fsync=True,directory_fsync=True,
    advisory_locking=True,attempt_private_staging=True)
repository = POSIXArtifactRepository(root / "repository",POSIXRepositoryPolicy(profile,
    maximum_chunk_bytes=1024,maximum_metadata_bytes=262144))
layout = AdaptiveSimplexLayout(3,3,vertex_capacity=32,cell_capacity=32,
    protected_edge_capacity=4,maximum_closure_iterations=32,maximum_coarsening_passes=2)
if stage == "save":
    rank = jax.process_index()
    points = np.asarray([[0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    rows = np.asarray([[0,1,2,5],[0,2,3,5],[0,3,4,5],[0,4,1,5]],np.int32)
    indices = np.arange(2*rank,2*rank+2,dtype=np.int32)
    ids = np.unique(rows[indices]).astype(np.int64)
    local = np.searchsorted(ids,rows[indices]).astype(np.int32)
    raw = adaptive_simplex_state(layout,coordinates=points[ids],vertex_ids=ids,
        vertex_active=np.ones(len(ids),np.bool_),vertex_parents=np.full((len(ids),2),-1,np.int32),
        vertex_protected=np.zeros(len(ids),np.bool_),cells=local,tuples=local,
        tags=np.full(2,3,np.int32),blocks=np.zeros(2,np.int32),generations=np.zeros(2,np.int32),
        parents=np.full(2,-1,np.int32),children=np.full((2,2),-1,np.int32),
        bisection_vertices=np.full(2,-1,np.int32),cell_ids=indices.astype(np.int64)+10,
        cell_active=np.ones(2,np.bool_),cell_classes=np.full(2,7,np.int32),
        facet_classes=np.zeros((2,4),np.int32),protected_edges=np.empty((0,2),np.int32),
        next_vertex_id=6,next_cell_id=14)
    parts = AdaptiveSimplexParts(jax.devices(),neighbor_pairs=((0,1),(1,0)))
    sharding = NamedSharding(parts.mesh,PartitionSpec(parts.axis_name))
    raw = jax.tree_util.tree_map(lambda value:jax.make_array_from_process_local_data(
        sharding,value[None],global_shape=(2,*value.shape)),raw)
    initial = raw
    first = refine_adaptive_simplex_parts(layout,parts,raw,raw.mesh.cell_active)
    undo = coarsen_adaptive_simplex_parts(layout,parts,first.state,first.state.mesh.cell_active)
    second = refine_adaptive_simplex_parts(layout,parts,undo.state,undo.state.mesh.cell_active)
    raw = second.state
    for report in (first.report,undo.report,second.report):
        assert np.all(np.asarray(report.status.addressable_shards[0].data)==0)
    raw = eqx.tree_at(lambda value:value.vertex_protected,raw,raw.vertex_removal>=0)
    codes = jnp.full((2,4),jnp.iinfo(jnp.int64).max,jnp.int64).at[0,0].set(1)
    raw = eqx.tree_at(lambda value:value.protected_codes,raw,codes)
    co = jnp.where(raw.mesh.cell_ids>=0,jnp.arange(2,dtype=jnp.int32)[:,None],-1)
    vo = jnp.full(raw.mesh.vertex_ids.shape,2,jnp.int32)
    for p in range(2):
        matches = jnp.any(raw.mesh.vertex_ids[:,:,None]==raw.mesh.vertex_ids[p,None,None,:],axis=-1)
        vo = jnp.minimum(vo,jnp.where(matches,p,2))
    vo = jnp.where(raw.mesh.vertex_ids>=0,vo,-1).astype(jnp.int32)
    arrays = {f"mesh/{name}":getattr(raw.mesh,name) for name in
        ("coordinates","vertex_ids","vertex_active","cells","cell_ids","cell_active","facet_neighbors")}
    state_names = ("vertex_half_facets","tuples","tags","blocks","generations","parents","children",
        "bisection_vertices","retired","cell_classes","facet_classes","vertex_parents","vertex_levels",
        "vertex_removal","vertex_protected","protected_codes","refine_rejected","coarsen_marked",
        "cursors","clocks","counters")
    arrays.update({name:getattr(raw,name) for name in state_names})
    arrays.update({f"initial/mesh/{name}":getattr(initial.mesh,name) for name in
        ("coordinates","vertex_ids","vertex_active","cells","cell_ids","cell_active","facet_neighbors")})
    arrays.update({f"initial/{name}":getattr(initial,name) for name in state_names})
    initial_co=jnp.where(initial.mesh.cell_ids>=0,jnp.arange(2,dtype=jnp.int32)[:,None],-1)
    initial_vo=jnp.full(initial.mesh.vertex_ids.shape,2,jnp.int32)
    for p in range(2):
        matches=jnp.any(initial.mesh.vertex_ids[:,:,None]==initial.mesh.vertex_ids[p,None,None,:],axis=-1)
        initial_vo=jnp.minimum(initial_vo,jnp.where(matches,p,2))
    arrays.update(initial_co=initial_co,initial_vo=jnp.where(initial.mesh.vertex_ids>=0,initial_vo,-1).astype(jnp.int32))
    arrays.update(co=co,vo=vo,cell_int=raw.mesh.cell_ids+2**54,cell_bool=raw.retired,
        cell_vector=raw.mesh.cell_ids[:,:,None]*jnp.asarray([2.,3.],jnp.float64),
        vertex_int=raw.mesh.vertex_ids+2**54,vertex_bool=raw.vertex_protected,
        vertex_vector=raw.mesh.coordinates)
    arrays = {name:jax.device_put(value,sharding) for name,value in arrays.items()}
    assert not raw.mesh.cell_ids.is_fully_addressable
    publish_process_checkpoint(repository,"forest","execution",arrays,
        analysis_plan_id="analysis",numeric_revision_id="revision",writer_id=f"writer-{rank}")
    multihost_utils.sync_global_devices("raw-forest-published")
    assemble_distributed_checkpoint_from_repository(repository,"forest","analysis","revision",
        "execution",expected_process_count=2)
    jax.distributed.shutdown()
else:
    count = int(sys.argv[3])
    fault = sys.argv[4]
    group = ExecutionRuntime.current().root_group
    placement = group.named_sharding(PartitionSpec())
    manifest = assemble_distributed_checkpoint_from_repository(repository,"forest","analysis",
        "revision","execution",expected_process_count=2)
    if fault == "packet_overflow":
        try:
            restore_global_array_from_checkpoint(repository,manifest,"['mesh/coordinates']",
                placement,maximum_host_packet_bytes=1)
        except ValueError as error:
            assert "maximum_host_packet_bytes" in str(error)
        else:
            raise RuntimeError("Undersized archive packet budget was accepted.")
        print("Immutable archive refused before numerical reconstruction")
        sys.exit(0)
    names = {dict(shard.metadata)["array_path"] for shard in manifest.shards}
    arrays = {path:restore_global_array_from_checkpoint(repository,manifest,path,placement,
        maximum_host_packet_bytes=1024) for path in names}
    def value(name: str) -> jax.Array:
        return arrays[jax.tree_util.keystr((jax.tree_util.DictKey(name),))]
    state_names = ("vertex_half_facets","tuples","tags","blocks","generations","parents","children",
        "bisection_vertices","retired","cell_classes","facet_classes","vertex_parents","vertex_levels",
        "vertex_removal","vertex_protected","protected_codes","refine_rejected","coarsen_marked",
        "cursors","clocks","counters")
    def restore_forest(prefix: str) -> AdaptiveSimplexState:
        mesh=jax.vmap(MaskedSimplexMesh)(*[value(f"{prefix}mesh/{name}") for name in
            ("coordinates","vertex_ids","vertex_active","cells","cell_ids","cell_active","facet_neighbors")])
        return jax.vmap(lambda mesh,**leaves:AdaptiveSimplexState(mesh,**leaves))(mesh,
            **{name:value(prefix+name) for name in state_names})
    raw=restore_forest("")
    co,vo = value("co"),value("vo")
    if fault == "duplicate":
        raw = eqx.tree_at(lambda x:x.mesh.cell_ids,raw,raw.mesh.cell_ids.at[1,0].set(raw.mesh.cell_ids[0,0]))
    elif fault == "missing":
        raw = eqx.tree_at(lambda x:x.mesh.vertex_ids,raw,raw.mesh.vertex_ids.at[:,0].set(-1))
    elif fault == "shared_conflict":
        raw = eqx.tree_at(lambda x:x.mesh.coordinates,raw,raw.mesh.coordinates.at[1,0,0].add(0.25))
    elif fault == "lost_authority":
        co = jnp.where(raw.mesh.cell_ids>=0,jnp.ones_like(co),-1)
    if fault == "overflow":
        layout = AdaptiveSimplexLayout(3,3,vertex_capacity=32,cell_capacity=8,
            protected_edge_capacity=4,maximum_closure_iterations=32,maximum_coarsening_passes=2)
    old = tuple(np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(raw))
    proposal = jnp.broadcast_to((jnp.arange(32,dtype=jnp.int32)%count)[None,:],(2,32))
    receipt = repack_simplex_restart(group,layout,raw,co,vo,proposal,
        cell_history=(value("cell_int"),value("cell_bool"),value("cell_vector")),
        vertex_history=(value("vertex_int"),value("vertex_bool"),value("vertex_vector")))
    for expected,leaf in zip(old,jax.tree_util.tree_leaves(raw),strict=True):
        np.testing.assert_array_equal(expected,leaf)
    if fault != "positive":
        required = AdaptiveSimplexStatus.CAPACITY_EXCEEDED if fault=="overflow" else AdaptiveSimplexStatus.INVALID_GEOMETRY
        assert np.all(np.asarray(receipt.status)&int(required))
        print("Immutable complete epoch refused:",fault)
    else:
        np.testing.assert_array_equal(receipt.status,np.zeros(count,np.int32))
        initial=restore_forest("initial/")
        targets,target_status=restart_forest_target_owners(initial.mesh.cell_ids,
            receipt.states.mesh.cell_ids,receipt.cell_owners)
        np.testing.assert_array_equal(target_status,0)
        initial_repack=repack_simplex_restart(group,layout,initial,value("initial_co"),
            value("initial_vo"),targets,cell_history=(initial.mesh.facet_neighbors<0,))
        np.testing.assert_array_equal(initial_repack.status,np.zeros(count,np.int32))
        assert int(jnp.sum(initial_repack.cells_after))==4
        _,missing_status=restart_forest_target_owners(initial.mesh.cell_ids.at[0,0].set(999),
            receipt.states.mesh.cell_ids,receipt.cell_owners)
        assert int(missing_status)&int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
        assert receipt.parts.part_count==count and receipt.states.mesh.cell_ids.shape==(count,32)
        assert len(receipt.parts.mesh.devices.flat)==count
        assert np.sum(np.asarray(receipt.cells_after))==20
        newids=np.asarray(receipt.states.mesh.cell_ids)
        used=newids>=0
        locations=np.asarray(receipt.cell_saved_locations)[used]
        for name in ("tags","blocks","generations","retired","cell_classes","facet_classes","refine_rejected","coarsen_marked"):
            np.testing.assert_array_equal(np.asarray(getattr(receipt.states,name))[used],
                np.asarray(getattr(raw,name))[locations[:,0],locations[:,1]])
        np.testing.assert_array_equal(np.asarray(receipt.cell_history[0])[used],newids[used]+2**54)
        np.testing.assert_array_equal(receipt.cell_history[1],receipt.states.retired)
        np.testing.assert_array_equal(np.asarray(receipt.cell_history[2])[used],newids[used,None]*[2.,3.])
        vids=np.asarray(receipt.states.mesh.vertex_ids)
        valid=vids>=0
        vlocations=np.asarray(receipt.vertex_saved_locations)[valid]
        for name in ("vertex_levels","vertex_removal","vertex_protected"):
            np.testing.assert_array_equal(np.asarray(getattr(receipt.states,name))[valid],
                np.asarray(getattr(raw,name))[vlocations[:,0],vlocations[:,1]])
        constraints=[]
        for p in range(count):
            codes=np.asarray(receipt.states.protected_codes[p])
            codes=codes[codes!=np.iinfo(np.int64).max]
            constraints.extend(vids[p,np.stack((codes//32,codes%32),axis=1)].tolist())
        assert constraints and all(edge==[0,1] for edge in constraints)
        np.testing.assert_array_equal(np.asarray(receipt.vertex_history[0])[valid],vids[valid]+2**54)
        np.testing.assert_array_equal(receipt.vertex_history[1],receipt.states.vertex_protected)
        np.testing.assert_array_equal(receipt.vertex_history[2],receipt.states.mesh.coordinates)
        np.testing.assert_array_equal(receipt.states.counters.sum(axis=0),raw.counters.sum(axis=0))
        continued=coarsen_adaptive_simplex_parts(layout,receipt.parts,receipt.states,receipt.states.mesh.cell_active)
        np.testing.assert_array_equal(continued.report.status,np.zeros(count,np.int32))
        active=np.asarray(continued.state.mesh.cell_active)
        np.testing.assert_array_equal(np.sort(np.asarray(continued.state.mesh.cell_ids)[active]),[10,11,12,13])
        print("Saved actual2process forest resumed on",count,"actual devices; typed history and coarsening preserved")
"""


@pytest.mark.parametrize(
    "scenario",
    (
        "refine",
        "coarsen",
        "mixed",
        "missing_owner",
        "incomplete_support",
        "contradictory_parent",
        "overlapping_source",
    ),
)
def test_solver_owner_inheritance_from_complete_scientific_forest(scenario: str) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    environment["PYTHONPATH"] = str(Path(phx.__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", _SOLVER_OWNER_INHERITANCE_SCRIPT, scenario],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + "\n" + completed.stderr, pytrace=False)


_SOLVER_OWNER_INHERITANCE_SCRIPT = r"""
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from phydrax.discretization._adaptive_simplex import (
    AdaptiveSimplexLayout, AdaptiveSimplexParts, AdaptiveSimplexState, AdaptiveSimplexStatus,
    adaptive_simplex_state, refine_adaptive_simplex_parts, coarsen_adaptive_simplex_parts,
)
from phydrax.meshing._distribution_forest import inherit_solver_cell_owners
scenario = sys.argv[1]
points = np.asarray([[0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
rows = np.asarray([[0,1,2,5],[0,2,3,5],[0,3,4,5],[0,4,1,5]], np.int32)
layout = AdaptiveSimplexLayout(3,3,vertex_capacity=64,cell_capacity=64,protected_edge_capacity=1,
    maximum_closure_iterations=64,maximum_coarsening_passes=1)
def piece(indices: np.ndarray) -> AdaptiveSimplexState:
    ids = np.unique(rows[indices]).astype(np.int64)
    local = np.searchsorted(ids,rows[indices]).astype(np.int32)
    count = indices.size
    return adaptive_simplex_state(layout,coordinates=points[ids],vertex_ids=ids,
        vertex_active=np.ones(ids.size,np.bool_),vertex_parents=np.full((ids.size,2),-1,np.int32),
        vertex_protected=np.zeros(ids.size,np.bool_),cells=local,tuples=local,
        tags=np.full(count,3,np.int32),blocks=np.zeros(count,np.int32),generations=np.zeros(count,np.int32),
        parents=np.full(count,-1,np.int32),children=np.full((count,2),-1,np.int32),
        bisection_vertices=np.full(count,-1,np.int32),cell_ids=indices.astype(np.int64)+10,
        cell_active=np.ones(count,np.bool_),cell_classes=np.full(count,7,np.int32),
        facet_classes=np.zeros((count,4),np.int32),protected_edges=np.empty((0,2),np.int32),
        next_vertex_id=6,next_cell_id=14)
source = jax.tree_util.tree_map(lambda *values:jnp.stack(values),
    piece(np.arange(2,dtype=np.int32)),piece(np.arange(2,4,dtype=np.int32)))
parts = AdaptiveSimplexParts(jax.devices(),neighbor_pairs=((0,1),(1,0)))
first = refine_adaptive_simplex_parts(layout,parts,source,source.mesh.cell_active)
np.testing.assert_array_equal(first.report.status,[0,0])
initial = first.state
# Solver siblings deliberately have different owners even within one execution family.
bank = jnp.where(initial.mesh.cell_active,initial.mesh.cell_ids%2,-1).astype(jnp.int32)
coarse = coarsen_adaptive_simplex_parts(layout,parts,initial,initial.mesh.cell_active)
np.testing.assert_array_equal(coarse.report.status,[0,0])
current = coarse.state
if scenario == "refine":
    update = refine_adaptive_simplex_parts(layout,parts,initial,initial.mesh.cell_active)
    np.testing.assert_array_equal(update.report.status,[0,0])
    current = update.state
elif scenario == "mixed":
    update = refine_adaptive_simplex_parts(layout,parts,coarse.state,coarse.state.mesh.cell_active)
    np.testing.assert_array_equal(update.report.status,[0,0])
    current = update.state
elif scenario == "missing_owner":
    bank = bank.at[1,2].set(-1)
elif scenario == "incomplete_support":
    initial = eqx.tree_at(lambda state:state.mesh.cell_active,initial,
        initial.mesh.cell_active.at[1,2].set(False))
    bank = bank.at[1,2].set(-1)
elif scenario == "contradictory_parent":
    initial = eqx.tree_at(lambda state:state.parents,initial,initial.parents.at[1,2].set(2))
elif scenario == "overlapping_source":
    initial = eqx.tree_at(lambda state:state.mesh.cell_active,initial,
        initial.mesh.cell_active.at[1,0].set(True))
    bank = bank.at[1,0].set(0)
owners,status = inherit_solver_cell_owners(parts,current,initial,bank)
if scenario in ("missing_owner","incomplete_support","contradictory_parent","overlapping_source"):
    assert np.all(np.asarray(status)&int(AdaptiveSimplexStatus.INVALID_GEOMETRY))
else:
    np.testing.assert_array_equal(status,[0,0])
    np.testing.assert_array_equal(np.asarray(owners)[~np.asarray(current.mesh.cell_active)],-1)
    if scenario == "refine":
        for rank in range(2):
            for slot in np.flatnonzero(np.asarray(current.mesh.cell_active[rank])):
                ancestor = slot
                while not bool(initial.mesh.cell_active[rank,ancestor]):
                    ancestor = int(current.parents[rank,ancestor])//2
                    assert ancestor>=0
                assert int(owners[rank,slot])==int(bank[rank,ancestor])
    else:
        # Both source children cover each restored region; canonical minimum is 0,
        # including a new branch created after coarsening but before publication.
        np.testing.assert_array_equal(np.asarray(owners)[np.asarray(current.mesh.cell_active)],0)
"""
