#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization._view_support import OwnerLocalCellGeometry
from phydrax.discretization.fem._distributed import (
    finite_element_dof_identity_plan,
    FiniteElementDofIdentityPlan,
)
from phydrax.discretization.fem._generic import FiniteElementDiscretization
from phydrax.meshing._assembly import MeshAssembly
from phydrax.meshing._contracts import SurfaceMeshingSpec


def _authority() -> phx.discretization.EmptyLocalCellMeshAuthority:
    return phx.discretization.EmptyLocalCellMeshAuthority(
        "accepted-global-topology",
        "accepted-global-geometry",
        "target-owner-directory",
        "execution-group",
        1,
        3,
        "triangles",
        "triangle",
        2,
        2,
    )


def test_empty_local_cell_mesh_requires_partition_authority() -> None:
    block = phx.discretization.CellBlock(
        "triangles", "triangle", np.empty((0, 3), dtype=np.int32)
    )
    with pytest.raises(ValueError, match="explicit partition authority"):
        phx.discretization.CellMesh(np.empty((0, 2)), (block,))

    authority = _authority()
    mesh = phx.discretization.CellMesh.empty_local(authority)
    assert mesh.coordinates.shape == (0, 2)
    assert mesh.blocks[0].cell_kind == "triangle"
    assert mesh.blocks[0].cell_count == 0
    assert mesh.empty_local_authority is not None
    assert mesh.empty_local_authority.authority_id == authority.authority_id

    refreshed = mesh.with_coordinates(mesh.coordinates, numeric_version="1")
    assert refreshed.topology_id == mesh.topology_id
    assert refreshed.geometry_id != mesh.geometry_id
    assert refreshed.empty_local_authority is not None
    assert refreshed.empty_local_authority.authority_id == authority.authority_id


def test_empty_local_finite_element_compilation_keeps_zero_resident_rows() -> None:
    mesh = phx.discretization.CellMesh.empty_local(_authority())
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.FiniteElementForm(
        "empty-local-mass-source",
        "u",
        (
            phx.equations.MassAction("u"),
            phx.equations.SourceAction("u", 1.0),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    residual = compiled.full_residual(jnp.empty((0,), dtype=jnp.float64))

    assert discretization.dof_maps[0].global_dof_count == 0
    assert discretization.dof_maps[0].dof_coordinates.shape == (0, 2)
    assert np.asarray(residual).shape == (0,)
    assert np.all(np.isfinite(np.asarray(residual)))

    empty_measure = phx.discretization.DiscreteMeasure(
        "empty-local-cells",
        "empty-local-support",
        "empty-local-cells",
        np.empty((0,), dtype=np.float64),
    )
    assert float(np.asarray(empty_measure.total_mass)) == 0.0
    assert bool(np.asarray(empty_measure.valid))
    with pytest.raises(ValueError, match="positive mass"):
        phx.discretization.DiscreteMeasure(
            "empty-local-probability",
            "empty-local-support",
            "empty-local-cells",
            np.empty((0,), dtype=np.float64),
            normalization="probability",
        )
    with pytest.raises(ValueError, match="positive mass"):
        phx.discretization.DiscreteMeasure(
            "zero-nonempty-physical",
            "nonempty-support",
            "nonempty-cells",
            np.zeros((1,), dtype=np.float64),
        )

    relation = phx.sparse.RowRelation(
        np.empty((0, 3), dtype=np.int32),
        source_size=0,
    )
    source_values = jnp.empty((0,), dtype=jnp.float64)
    routed = phx.sparse.gather_routes(relation, source_values)
    reduced = phx.sparse.route_reduce(relation, routed)
    linear = phx.sparse.SparseLinearMap(relation, np.empty((0, 3), dtype=np.float64))
    assert routed.shape == (0, 3)
    assert reduced.shape == (0,)
    assert linear.mv(source_values).shape == (0,)
    assert linear.transpose_mv(source_values).shape == (0,)
    assert linear.as_dense().shape == (0, 0)
    with pytest.raises(ValueError, match="outside \\[0, 0\\)"):
        phx.sparse.RowRelation(np.zeros((1, 3), dtype=np.int32), source_size=0)
    padded = phx.sparse.RowRelation(
        np.zeros((2, 3), dtype=np.int32),
        source_size=0,
        valid=np.zeros((2, 3), dtype=np.bool_),
    )
    padded_linear = phx.sparse.SparseLinearMap(padded, np.zeros((2, 3), dtype=np.float64))
    padded_values = padded_linear.mv(source_values)
    assert padded_values.shape == (2,)
    np.testing.assert_array_equal(padded_values, np.zeros((2,), dtype=np.float64))
    assert padded_linear.transpose_mv(padded_values).shape == (0,)
    assert padded_linear.as_dense().shape == (2, 0)


def _native_periodic_surface(
    name: str,
    fractional_points: np.ndarray,
    point_owners: np.ndarray,
    group: ExecutionGroup,
) -> tuple[MeshAssembly, SurfaceMeshingSpec]:
    import jax

    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        SurfaceMeshingSpec,
    )
    from phydrax.meshing._periodic import PeriodicPointOrbits
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl
    from phydrax.meshing.providers._native_options import NativeMeshingOptions
    from phydrax.meshing.providers._native_periodic import (
        NativePeriodicSource,
        prepare_native_periodic,
    )

    revision = f"{name}-revision-1"
    cell = PeriodicCell(np.asarray(((1.0, 0.2), (0.3, 1.1)), dtype=np.float64))
    point_ids = np.asarray((91, 4, 53, 17), dtype=np.int64)
    local_rows = point_owners == jax.process_index()
    source = NativePeriodicSource(
        PeriodicPointOrbits(
            cell,
            np.asarray(cell.cartesian(fractional_points[local_rows]), dtype=np.float64),
            point_ids[local_rows],
        ),
        name,
        revision,
        point_owner_ranks=point_owners[local_rows],
    )
    domain = MeshingScope(
        name,
        revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
        domain,
        size_controls=(UniformSizeControl(domain, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    assembly = prepare_native_periodic(
        source,
        specification,
        NativeMeshingOptions("periodic_delaunay"),
        phx.SpatialCoordinateContract.si(),
    ).execute(execution_group=group)
    return assembly, specification


def _compile_owner_local_problem(
    view: OwnerLocalCellGeometry,
    group: ExecutionGroup,
    rank: int,
    /,
) -> tuple[FiniteElementDiscretization, Any, FiniteElementDofIdentityPlan, Any]:

    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("Owner-local FE compilation requires OwnerLocalCellGeometry.")
    mesh, coordinate_spec = view.reconstruct_local_geometry()
    if mesh is None:
        # An empty owner carries no blocks; the periodic Delaunay producer names
        # its single simplex block "periodic" on every owner.
        authority = phx.discretization.EmptyLocalCellMeshAuthority(
            view.source_topology_id,
            view.source_geometry_id,
            view.distribution_id,
            group.spec.group_id,
            rank,
            view.owner_directory_data.part_count,
            view.blocks[0].name if view.blocks else "periodic",
            view.blocks[0].cell_kind if view.blocks else "triangle",
            view.topological_dimension,
            view.coordinates.shape[1],
        )
        mesh = phx.discretization.CellMesh.empty_local(authority)
    local_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    if not np.array_equal(local_cell_ids, np.asarray(view.cell_ids, dtype=np.int64)):
        raise ValueError("FE carrier rows differ from the accepted owner-local view.")
    field_spec = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    plan = (
        phx.discretization.FiniteElementPlan(
            mesh, field_spec, coordinate_spec=coordinate_spec
        )
        if coordinate_spec is not None
        else phx.discretization.FiniteElementPlan(mesh, field_spec)
    )
    discretization = plan.prepare()
    form = phx.equations.FiniteElementForm(
        "owner-local-empty-mass-source",
        "u",
        (
            phx.equations.MassAction("u"),
            phx.equations.SourceAction("u", 1.0),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    field_layout = phx.solver.FiniteElementAcceptedFieldLayout(
        field_spec,
        transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    identity = finite_element_dof_identity_plan(
        discretization,
        0,
        field_layout_id=field_layout.layout_id,
        distribution_id=view.distribution_id,
        topology_id=view.source_topology_id,
        geometry_id=view.source_geometry_id,
        local_rank=rank,
        part_count=view.owner_directory_data.part_count,
        cell_ids=view.cell_ids,
        cell_owner_ranks=view.cell_owner_ranks,
        vertex_ids=view.vertex_ids,
        vertex_owner_ranks=view.vertex_owner_ranks,
    )
    return discretization, compiled, identity, field_layout


def _native_owner_local_fe_empty_rank_worker() -> None:
    import json

    import jax
    from jax.experimental import multihost_utils

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._assembly import MeshPartManifest
    from phydrax.meshing._distribution import OwnerLocalGraphMigrationEvidence
    from phydrax.meshing.providers._native_periodic import (
        _repartition_owner_local_periodic_assembly,
    )
    from phydrax.solver._finite_element_adaptivity import (
        FiniteElementTopologyTransaction,
        partition_owner_local_finite_element_graph,
    )

    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    ranks = tuple(device.process_index for device in devices)
    if (
        jax.process_count() != 3
        or ranks != (0, 1, 2)
        or len(devices) != 3
        or len(jax.local_devices()) != 1
    ):
        raise RuntimeError("The FE ownership worker requires three real CPU owners.")
    rank = jax.process_index()
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "native-owner-local-empty-fe",
            ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("native_owner_local_empty_fe", 3),),
        ),
        devices,
    )
    points = np.asarray(
        ((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77)),
        dtype=np.float64,
    )
    point_owners = np.zeros((points.shape[0],), dtype=np.int32)
    source_assembly, specification = _native_periodic_surface(
        "native-owner-local-empty-fe",
        points,
        point_owners,
        group,
    )
    source_view = source_assembly.parts[0].carrier
    if not isinstance(source_view, OwnerLocalCellGeometry):
        raise TypeError("Native source assembly did not return an owner-local geometry.")
    (
        source_discretization,
        source_compiled,
        source_identity,
        field_layout,
    ) = _compile_owner_local_problem(source_view, group, rank)
    graph_plan = GraphPartitionPlan(
        3,
        target_shares=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        maximum_imbalance=1.0,
        require_nonempty=False,
        refinement_passes=1,
    )
    graph_evidence = partition_owner_local_finite_element_graph(
        source_view,
        source_compiled,
        graph_plan,
        group,
        axis_name="native_owner_local_empty_fe_graph",
    )
    target_assembly = _repartition_owner_local_periodic_assembly(
        source_assembly,
        graph_evidence,
        group,
        current_specification=specification,
        graph_partition_plan=graph_plan,
        axis_name="native_owner_local_empty_fe_geometry",
    )
    target_view = target_assembly.parts[0].carrier
    if not isinstance(target_view, OwnerLocalCellGeometry):
        raise TypeError("Native target assembly did not return an owner-local geometry.")
    (
        target_discretization,
        target_compiled,
        target_identity,
        target_field_layout,
    ) = _compile_owner_local_problem(target_view, group, rank)
    if (
        source_identity.field_layout_id != field_layout.layout_id
        or target_identity.field_layout_id != target_field_layout.layout_id
        or field_layout.layout_id != target_field_layout.layout_id
    ):
        raise ValueError("FE identity plans differ from the accepted scalar field.")

    source_keys = np.asarray(source_identity.dof_keys, dtype=np.int64)
    source_full_values = np.asarray(
        0.25 + 0.01 * source_keys[:, 1] + 0.001 * source_keys[:, 2],
        dtype=np.float64,
    )
    source_values, source_ownership = source_identity.shard_values(source_full_values)
    site = phx.equations.MaterialSiteId("steel:integration-site:0")
    material = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                site,
                "linear-elastic:steel",
                np.asarray((2.5, 3.5), dtype=np.float64),
                trial=np.asarray((2.75, 3.75), dtype=np.float64),
                state_version=3,
            ),
        )
    )
    state = phx.solver.FiniteElementAcceptedState(
        (source_values,),
        np.asarray(0.125, dtype=np.float64),
        7,
        source_view.source_topology_id,
        source_discretization.prepared_id,
        source_compiled.compilation_id,
        field_layouts=(field_layout,),
        materials=material,
        field_ownership=(source_ownership,),
    )
    history_name = "accepted-history"
    source_history = source_values + 2.0
    source_history_layout = phx.solver.FiniteElementRestartStateLayout(
        history_name,
        phx.solver.FiniteElementRestartStateRole.HISTORY,
        phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
        ownership=source_ownership,
    )
    previous_name = "previous-step"
    rng_name = "rng"
    rng = jax.random.key(2026)
    previous_layout = phx.solver.FiniteElementRestartStateLayout(
        previous_name,
        phx.solver.FiniteElementRestartStateRole.HISTORY,
        phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
    )
    rng_layout = phx.solver.FiniteElementRestartStateLayout(
        rng_name,
        phx.solver.FiniteElementRestartStateRole.RNG,
        phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
        prng_impl=str(jax.random.key_impl(rng)),
    )
    material_layout = phx.solver.FiniteElementRestartStateLayout(
        site.key,
        phx.solver.FiniteElementRestartStateRole.MODEL_STATE,
        phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
    )
    source_manifest = phx.solver.FiniteElementRestartManifest(
        state,
        auxiliary_state=((history_name, source_history),),
        integrator_state=(
            (previous_name, np.asarray(0.125, dtype=np.float64)),
            (rng_name, rng),
        ),
        auxiliary_layouts=(source_history_layout,),
        integrator_layouts=(previous_layout, rng_layout),
        material_layouts=(material_layout,),
    )

    source_part = source_assembly.parts[0]
    target_part = target_assembly.parts[0]
    source_geometry_manifest = MeshPartManifest.from_part(source_part)
    target_geometry_manifest = MeshPartManifest.from_part(target_part)
    if source_view.source_value is None:
        raise ValueError(
            "Native owner-local FE source lacks its root geometry authority."
        )
    source_root_entry = phx.lifecycle.CompositionEntry(
        source_view.source_value,
        entry_id="mesh/source-root",
        role="topology",
        owner_id="test-owner-local-fe-source",
        structure_id=source_view.source_mesh_id,
        revision_id=source_view.source_revision,
        semantics_id=source_view.source_geometry_id,
    )
    mesh_entry = phx.lifecycle.CompositionEntry(
        source_geometry_manifest,
        entry_id="mesh/topology",
        role="topology",
        owner_id="test-owner-local-fe",
        structure_id=source_view.source_topology_id,
        revision_id=source_geometry_manifest.manifest_id,
        semantics_id="cell-mesh",
    )
    geometry_entry = phx.lifecycle.CompositionEntry(
        source_geometry_manifest,
        entry_id="mesh/geometry",
        role="discretization",
        owner_id="test-owner-local-fe",
        structure_id=source_view.source_geometry_id,
        revision_id=source_geometry_manifest.manifest_id,
        semantics_id="compact-owner-local-geometry",
        dependencies=(
            mesh_entry.binding("structure"),
            source_root_entry.binding("structure"),
            source_root_entry.binding("revision"),
            source_root_entry.binding("semantics"),
        ),
    )
    state_entry = phx.lifecycle.CompositionEntry(
        state,
        entry_id="solver/state",
        role="physical-state",
        owner_id="test-owner-local-fe-solver",
        structure_id=source_view.source_topology_id,
        revision_id=state.accepted_id,
        semantics_id="accepted-fe-state",
        dependencies=(
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
        ),
    )
    field_entry = phx.lifecycle.CompositionEntry(
        source_values,
        entry_id="solver/state/field/u",
        role="physical-state",
        owner_id=state_entry.owner_id,
        structure_id=state.topology_id,
        revision_id=state.accepted_id,
        semantics_id=field_layout.layout_id,
        dependencies=(state_entry.binding("revision"),),
    )
    history_entry = phx.lifecycle.CompositionEntry(
        source_history,
        entry_id=f"solver/restart/auxiliary/{history_name}",
        role=source_history_layout.role.value,
        owner_id=state_entry.owner_id,
        structure_id=source_history_layout.layout_id,
        revision_id=source_manifest.manifest_id,
        semantics_id=source_history_layout.semantics_id,
        dependencies=(state_entry.binding("structure"),),
    )
    previous_value = np.asarray(0.125, dtype=np.float64)
    previous_entry = phx.lifecycle.CompositionEntry(
        previous_value,
        entry_id=f"solver/restart/integrator/{previous_name}",
        role=previous_layout.role.value,
        owner_id=state_entry.owner_id,
        structure_id=previous_layout.layout_id,
        revision_id=source_manifest.manifest_id,
        semantics_id=previous_layout.semantics_id,
        dependencies=(state_entry.binding("structure"),),
    )
    rng_entry = phx.lifecycle.CompositionEntry(
        rng,
        entry_id=f"solver/restart/integrator/{rng_name}",
        role=rng_layout.role.value,
        owner_id=state_entry.owner_id,
        structure_id=rng_layout.layout_id,
        revision_id=source_manifest.manifest_id,
        semantics_id=rng_layout.semantics_id,
        dependencies=(state_entry.binding("structure"),),
    )
    material_entry = phx.lifecycle.CompositionEntry(
        material,
        entry_id="solver/state/materials",
        role="model-state",
        owner_id=state_entry.owner_id,
        structure_id=material.layout_id,
        revision_id=material.transaction_id,
        semantics_id="replicated-material-transaction",
        dependencies=(state_entry.binding("structure"),),
    )
    source_composition = phx.lifecycle.Composition(
        (
            source_root_entry,
            mesh_entry,
            geometry_entry,
            state_entry,
            field_entry,
            history_entry,
            previous_entry,
            rng_entry,
            material_entry,
        ),
        boundary_id="owner-local-fe-accepted",
    )
    source_composition_id = source_composition.composition_id
    if target_view.source_value is None:
        raise ValueError(
            "Native owner-local FE target lacks its root geometry authority."
        )
    target_root_entry = source_root_entry
    target_root_reprepared: tuple[Any, ...] = ()
    if target_view.source_value is not source_view.source_value:
        target_root_entry = phx.lifecycle.CompositionEntry(
            target_view.source_value,
            entry_id=source_root_entry.entry_id,
            role=source_root_entry.role,
            owner_id=source_root_entry.owner_id,
            structure_id=target_view.source_mesh_id,
            revision_id=target_view.source_revision,
            semantics_id=target_view.source_geometry_id,
        )
        target_root_reprepared = (target_root_entry,)
    target_mesh_entry = phx.lifecycle.CompositionEntry(
        target_geometry_manifest,
        entry_id=mesh_entry.entry_id,
        role=mesh_entry.role,
        owner_id=mesh_entry.owner_id,
        structure_id=target_view.source_topology_id,
        revision_id=target_geometry_manifest.manifest_id,
        semantics_id=mesh_entry.semantics_id,
    )
    target_geometry_entry = phx.lifecycle.CompositionEntry(
        target_geometry_manifest,
        entry_id=geometry_entry.entry_id,
        role=geometry_entry.role,
        owner_id=geometry_entry.owner_id,
        structure_id=target_view.source_geometry_id,
        revision_id=target_geometry_manifest.manifest_id,
        semantics_id=geometry_entry.semantics_id,
        dependencies=(
            target_mesh_entry.binding("structure"),
            target_root_entry.binding("structure"),
            target_root_entry.binding("revision"),
            target_root_entry.binding("semantics"),
        ),
    )
    target_reprepared_entries = (
        *target_root_reprepared,
        target_mesh_entry,
        target_geometry_entry,
    )
    migration = target_view.graph_migration_evidence
    if not isinstance(migration, OwnerLocalGraphMigrationEvidence):
        raise TypeError("Target owner view lacks graph migration evidence.")
    target_history_layout = phx.solver.FiniteElementRestartStateLayout(
        history_name,
        phx.solver.FiniteElementRestartStateRole.HISTORY,
        phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
        ownership=target_identity.ownership(),
    )
    target_transaction = FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: True
    )

    def repartition(
        auxiliary_layout: phx.solver.FiniteElementRestartStateLayout,
    ) -> phx.solver.FiniteElementRepartitionResult:
        return target_transaction.repartition(
            source_manifest,
            source_compiled,
            target_compiled,
            (source_identity,),
            (target_identity,),
            migration,
            source_composition,
            group,
            source_view,
            target_view,
            state_entry_id=state_entry.entry_id,
            geometry_entry_id=geometry_entry.entry_id,
            field_entry_ids=(("u", field_entry.entry_id),),
            material_entry_id=material_entry.entry_id,
            auxiliary_entry_ids=((history_name, history_entry.entry_id),),
            integrator_entry_ids=(
                (previous_name, previous_entry.entry_id),
                (rng_name, rng_entry.entry_id),
            ),
            target_auxiliary_layouts=(auxiliary_layout,),
            target_integrator_layouts=(previous_layout, rng_layout),
            target_material_layouts=(material_layout,),
            # Materials are transported; graph migration keeps the authored
            # source root, and only a changed root is re-prepared.
            retain_entry_ids=(
                () if target_root_reprepared else (source_root_entry.entry_id,)
            ),
            reprepare_entries=target_reprepared_entries,
            axis_name="native_owner_local_empty_fe_repartition",
        )

    refused = repartition(
        phx.solver.FiniteElementRestartStateLayout(
            "invalid-history",
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
            ownership=target_identity.ownership(),
        )
        if rank == 1
        else target_history_layout
    )
    if (
        bool(np.asarray(refused.committed))
        or refused.state.accepted_id != state.accepted_id
    ):
        raise ValueError("Collective FE refusal changed an accepted source state.")
    if source_composition.composition_id != source_composition_id:
        raise ValueError("Collective FE refusal mutated its source composition.")

    result = repartition(target_history_layout)
    if not bool(np.asarray(result.committed)) or result.rebind_receipt is None:
        raise ValueError(f"Owner-local FE repartition refused: {result.diagnostics}")

    target_owned_keys = np.asarray(target_identity.ownership().dof_keys, dtype=np.int64)
    expected_target_values = np.asarray(
        0.25 + 0.01 * target_owned_keys[:, 1] + 0.001 * target_owned_keys[:, 2],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(result.state.fields[0]), expected_target_values)
    target_history = dict(result.manifest.auxiliary_state)[history_name]
    np.testing.assert_allclose(np.asarray(target_history), expected_target_values + 2.0)
    target_integrator = dict(result.manifest.integrator_state)
    np.testing.assert_array_equal(target_integrator[previous_name], previous_value)
    target_rng = target_integrator[rng_name]
    if not jax.dtypes.issubdtype(target_rng.dtype, jax.dtypes.prng_key):
        raise ValueError("Typed FE RNG was not preserved by repartition.")
    np.testing.assert_array_equal(
        jax.random.key_data(target_rng), jax.random.key_data(rng)
    )
    target_materials = result.state.materials
    if target_materials is None:
        raise ValueError("Replicated material transaction was lost.")
    source_material_state = material.state(site)
    target_material_state = target_materials.state(site)
    if (
        target_material_state.model_id != source_material_state.model_id
        or target_material_state.state_version != source_material_state.state_version
    ):
        raise ValueError("Replicated material identity changed during repartition.")
    np.testing.assert_array_equal(
        target_material_state.committed, source_material_state.committed
    )
    np.testing.assert_array_equal(
        target_material_state.trial, source_material_state.trial
    )
    if (
        not result.rebind_receipt.published
        or result.rebind_receipt.composition.entry(geometry_entry.entry_id).revision_id
        != target_geometry_manifest.manifest_id
    ):
        raise ValueError(
            "Published FE state is not bound to the current target geometry."
        )
    target_fields = result.field_transfers[0].gather_target_local_values(
        target_identity.dof_keys,
        group,
        axis_name="native_owner_local_empty_fe_target_values",
    )
    residual = target_compiled.full_residual(target_fields)
    if not np.all(np.isfinite(np.asarray(residual))):
        raise ValueError("Target compiled FE consumer produced non-finite residuals.")
    target_cell_count = sum(
        block.cell_count for block in target_discretization.mesh.blocks
    )
    local_solve = 0
    if target_discretization.dof_maps[0].global_dof_count:
        operator, right_hand_side = target_compiled.linear_system()
        solution = phx.linalg.solve(operator, right_hand_side)
        if not bool(np.all(np.asarray(solution.successful))):
            raise ValueError("Target FE continuation solve did not converge.")
        local_solve = 1
    counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    rank,
                    int(source_view.owned_cell_ids.size),
                    target_cell_count,
                    int(target_view.owned_cell_ids.size),
                    int(target_discretization.dof_maps[0].global_dof_count),
                    int(target_discretization.mesh.coordinates.shape[0]),
                    int(np.count_nonzero(~np.asarray(target_view.owned, dtype=np.bool_))),
                    local_solve,
                ),
                dtype=np.int64,
            )
        )
    )
    print(
        json.dumps(
            {
                "phase": "committed",
                "rank": rank,
                "rows": counts.tolist(),
                "target_step": int(np.asarray(result.state.step)),
            },
            sort_keys=True,
        )
    )


def _launch_native_owner_local_fe_worker(worker_name: str, /) -> list[dict[str, Any]]:
    import json
    import os
    import socket
    import sys
    from pathlib import Path

    import jax

    from phydrax import config as pxconfig
    from phydrax._execution_launch import (
        launch_local_processes,
        LocalProcessLaunchError,
        LocalProcessLaunchPlan,
    )

    if jax.process_count() != 1:
        pytest.skip("The three-owner FE coordinator must run outside a live group.")
    if os.name != "posix" or os.environ.get("PYTEST_XDIST_WORKER") is not None:
        pytest.skip(
            "The native FE ownership fixture requires POSIX supervision outside xdist."
        )
    repository = Path(__file__).resolve().parents[3]
    worker_script = f"import runpy, sys; runpy.run_path(sys.argv[1])[{worker_name!r}]()"
    base_environment = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        base_environment.pop(name, None)
    library = base_environment.get("PHYDRAX_MESHCORE_LIBRARY")
    if library:
        base_environment["PHYDRAX_MESHCORE_LIBRARY"] = str(
            Path(library).resolve(strict=True)
        )
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_device_count=1,
            cpu_collectives_implementation="gloo",
        ),
        base=base_environment,
    )
    environment.update(
        PYTHONPATH=str(repository),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        results = launch_local_processes(
            (
                sys.executable,
                "-c",
                worker_script,
                str(Path(__file__).resolve()),
            ),
            LocalProcessLaunchPlan(3, f"127.0.0.1:{port}", devices_per_process=1),
            environment=environment,
            cwd=str(repository),
            timeout_seconds=900.0,
        )
    except LocalProcessLaunchError as error:
        # The launcher keeps only stderr tails; the owning Python traceback
        # precedes the coordination-service shutdown noise.
        details = "\n".join(
            f"rank {result.process_id} exited {result.returncode}; full stderr:\n"
            + result.stderr.decode("utf-8", errors="replace")
            for result in error.results
        )
        raise AssertionError(
            f"{error.reason or 'FE worker launch failed'}\n{details}"
        ) from error
    return [
        json.loads(result.stdout.decode().strip().splitlines()[-1]) for result in results
    ]


def test_native_owner_local_fe_repartition_supports_real_empty_owner() -> None:
    """Move actual native FE cells across three ranks, including zero residents."""
    reports = _launch_native_owner_local_fe_worker(
        "_native_owner_local_fe_empty_rank_worker"
    )
    assert [report["rank"] for report in reports] == [0, 1, 2]
    assert all(report["phase"] == "committed" for report in reports)
    rows = np.asarray(reports[0]["rows"], dtype=np.int64)
    assert rows.shape == (3, 8)
    assert rows[0, 1] > 0
    assert rows[1, 1] == 0 and rows[2, 1] == 0
    assert rows[0, 2] == 0 and rows[1, 2] == 0
    assert rows[0, 3] == 0 and rows[1, 3] == 0 and rows[2, 3] > 0
    assert rows[0, 4] == 0 and rows[1, 4] == 0 and rows[2, 4] > 0
    assert rows[0, 5] == 0 and rows[1, 5] == 0 and rows[2, 5] > 0
    assert rows[:, 6].tolist() == [0, 0, 0]
    assert rows[:, 7].tolist() == [0, 0, 1]


def test_native_owner_local_fe_adaptation_transfers_and_solves() -> None:
    """Adapt typed P1/P0 state and restart banks, then move it to new owners."""
    reports = _launch_native_owner_local_fe_worker(
        "_native_owner_local_fe_topology_worker"
    )
    assert [report["rank"] for report in reports] == [0, 1, 2]
    assert all(report["phase"] == "adapted" for report in reports)
    rows = np.asarray(reports[0]["rows"], dtype=np.int64)
    assert rows.shape == (3, 7)
    # Adaptation keeps the source placement: ranks 0 and 1 own cells, rank 2 none.
    assert rows[0, 0] > 0 and rows[1, 0] > 0 and rows[2, 0] == 0
    assert rows[0, 1] > 0 and rows[1, 1] > 0 and rows[2, 1] == 0
    # The repartition moves every cell onto rank 2; P1 rows follow vertex owners.
    assert rows[0, 2] == 0 and rows[1, 2] == 0 and rows[2, 2] > 0
    assert rows[2, 3] > 0
    assert np.all(rows[:, 4:] == 1)


def _native_owner_local_fe_topology_worker() -> None:
    import json

    import equinox as eqx
    import jax
    import numpy as np
    from jax.experimental import multihost_utils

    import phydrax as phx
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.lifecycle import CompositionEntry
    from phydrax.meshing._assembly import MeshPartManifest
    from phydrax.meshing._distribution import OwnerLocalGraphMigrationEvidence
    from phydrax.solver._finite_element_adaptivity import (
        FiniteElementTopologyTransaction,
    )
    from tools.meshing_qualification.scenarios.cold_owner_local import (
        _compile_owner_local_problem,
        _current_group,
        _owner_local_views,
        _periodic_p0_owner_integral,
        _solve_physical_problem,
        _source_composition,
    )

    group, rank = _current_group()
    (
        source_assembly,
        adaptation_assembly,
        target_assembly,
        source_view,
        adaptation_view,
        target_view,
        specification,
        adaptation_specification,
        source_graph_plan,
        target_graph_plan,
        _,
        target_adaptive_result,
        adaptive_result,
    ) = _owner_local_views(group, "routine")
    # The adaptation runs on the source placement (rank 2 holds no cells); the
    # adapted state is then repartitioned onto the target placement (only rank
    # 2 holds cells), whose view keeps the rebound adaptation provenance.
    if (
        not isinstance(source_view, OwnerLocalCellGeometry)
        or not isinstance(adaptation_view, OwnerLocalCellGeometry)
        or not isinstance(target_view, OwnerLocalCellGeometry)
        or adaptation_view.adaptive_result is not adaptive_result
        or target_view.adaptive_result is not target_adaptive_result
        or target_adaptive_result.executor_result_id != adaptive_result.executor_result_id
        or (adaptive_result.vertex_stencil is None)
        != (adaptation_view.owned_cell_ids.size == 0)
        or (adaptive_result.lineage is None) != (adaptation_view.owned_cell_ids.size == 0)
    ):
        raise TypeError("The periodic producer omitted current adaptive FE authority.")

    # Restore must refuse issuer/carrier objects rather than treating them as source data.
    bad_source = eqx.tree_at(
        lambda view: view.source_value,
        source_view,
        CompositionEntry(
            "not-a-source",
            entry_id="invalid/source",
            role="topology",
            owner_id="invalid-test",
            structure_id="invalid-structure",
            revision_id="invalid-revision",
            semantics_id="invalid-semantics",
        ),
    )
    try:
        OwnerLocalCellGeometry.from_stored_fields(
            bad_source,
            group,
            current_specification=specification,
            axis_name="native_fe_bad_source_value",
        )
    except (TypeError, ValueError) as error:
        # Stored-view restore refuses collectively, so the owner's TypeError
        # reaches every rank inside the group refusal message.
        if "actual NativePeriodicSource data" not in str(error):
            raise
    else:
        raise AssertionError("Stored owner-local source accepted a CompositionEntry.")
    bad_authority = eqx.tree_at(
        lambda view: view.source_geometry_authority,
        source_view,
        source_view,
    )
    try:
        OwnerLocalCellGeometry.from_stored_fields(
            bad_authority,
            group,
            current_specification=specification,
            axis_name="native_fe_bad_source_authority",
        )
    except ValueError as error:
        if "Nested owner-local source views" not in str(error):
            raise
    else:
        raise AssertionError("Stored source authority accepted a nested local view.")

    (
        source_discretization,
        source_compiled,
        source_identities,
        source_layouts,
    ) = _compile_owner_local_problem(source_view, group, rank)
    (
        adapted_discretization,
        adapted_compiled,
        adapted_identities,
        adapted_layouts,
    ) = _compile_owner_local_problem(adaptation_view, group, rank)
    (
        target_discretization,
        target_compiled,
        target_identities,
        target_layouts,
    ) = _compile_owner_local_problem(target_view, group, rank)
    if (
        len(source_identities) != 2
        or len(adapted_identities) != 2
        or len(target_identities) != 2
        or tuple(layout.layout_id for layout in source_layouts)
        != tuple(layout.layout_id for layout in adapted_layouts)
        or tuple(layout.layout_id for layout in source_layouts)
        != tuple(layout.layout_id for layout in target_layouts)
        or tuple(layout.field_spec.name for layout in source_layouts) != ("u", "content")
    ):
        raise ValueError("The adaptive FE plans do not carry the declared P1/P0 fields.")

    source_physical = _solve_physical_problem(
        source_compiled,
        tuple(
            int(dof_map.global_dof_count) for dof_map in source_discretization.dof_maps
        ),
    )
    source_full_fields = (
        np.asarray(source_physical["values"], dtype=np.float64),
        np.asarray(source_physical["content_values"], dtype=np.float64),
    )
    source_shards = tuple(
        identity.shard_values(values)
        for identity, values in zip(source_identities, source_full_fields, strict=True)
    )
    source_fields = tuple(values for values, _ in source_shards)
    source_ownership = tuple(ownership for _, ownership in source_shards)
    if source_fields[0].size and not np.all(np.asarray(source_fields[0]) > 0.0):
        raise ValueError("The accepted P1 source field is not positive.")

    material_site = phx.equations.MaterialSiteId("steel:integration-site:0")
    materials = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                material_site,
                "linear-elastic:steel",
                np.asarray((2.5, 3.5), dtype=np.float64),
                trial=np.asarray((2.75, 3.75), dtype=np.float64),
                state_version=3,
            ),
        )
    )
    source_state = phx.solver.FiniteElementAcceptedState(
        source_fields,
        np.asarray(0.125, dtype=np.float64),
        7,
        source_view.source_topology_id,
        source_discretization.prepared_id,
        source_compiled.compilation_id,
        field_layouts=source_layouts,
        materials=materials,
        schedule_cursor=4,
        state_version=9,
        field_ownership=source_ownership,
    )
    u_history_name = "u-history"
    content_history_name = "content-history"
    clock_history_name = "clock-history"
    previous_name = "previous-step"
    rng_name = "rng"
    source_auxiliary_layouts = (
        phx.solver.FiniteElementRestartStateLayout(
            u_history_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
            ownership=source_ownership[0],
        ),
        phx.solver.FiniteElementRestartStateLayout(
            content_history_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
            ownership=source_ownership[1],
        ),
        phx.solver.FiniteElementRestartStateLayout(
            clock_history_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
        ),
    )
    rng = jax.random.key(2026)
    integrator_layouts = (
        phx.solver.FiniteElementRestartStateLayout(
            previous_name,
            phx.solver.FiniteElementRestartStateRole.HISTORY,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
        ),
        phx.solver.FiniteElementRestartStateLayout(
            rng_name,
            phx.solver.FiniteElementRestartStateRole.RNG,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
            prng_impl=str(jax.random.key_impl(rng)),
        ),
    )
    material_layouts = (
        phx.solver.FiniteElementRestartStateLayout(
            material_site.key,
            phx.solver.FiniteElementRestartStateRole.MODEL_STATE,
            phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
        ),
    )
    clock_history = np.asarray((0.0, 0.125), dtype=np.float64)
    previous_step = np.asarray(0.125, dtype=np.float64)
    restart_manifest = phx.solver.FiniteElementRestartManifest(
        source_state,
        auxiliary_state=(
            (u_history_name, source_fields[0] + 2.0),
            (content_history_name, source_fields[1] + 0.375),
            (clock_history_name, clock_history),
        ),
        integrator_state=((previous_name, previous_step), (rng_name, rng)),
        auxiliary_layouts=source_auxiliary_layouts,
        integrator_layouts=integrator_layouts,
        material_layouts=material_layouts,
    )
    source_geometry_manifest = MeshPartManifest.from_part(source_assembly.parts[0])
    source_composition = _source_composition(
        source_view,
        source_geometry_manifest,
        restart_manifest,
        source_layouts,
        specification,
        adaptation_specification,
        source_graph_plan,
        target_graph_plan,
    )
    source_composition_id = source_composition.composition_id
    source_root_entry = source_composition.entry("mesh/source")

    def placement_entries(assembly: Any, /) -> tuple[Any, Any]:
        manifest = MeshPartManifest.from_part(assembly.parts[0])
        topology_id = manifest.source_topology_id
        geometry_id = manifest.source_geometry_id
        if topology_id is None or geometry_id is None:
            raise ValueError(
                "Current periodic placement lacks canonical topology/geometry IDs."
            )
        topology_entry = CompositionEntry(
            manifest,
            entry_id="mesh/topology",
            role="topology",
            owner_id="phydrax.meshing",
            structure_id=topology_id,
            revision_id=manifest.manifest_id,
            semantics_id="cell-mesh",
            dependencies=(
                source_root_entry.binding("structure"),
                source_root_entry.binding("revision"),
                source_root_entry.binding("semantics"),
                source_composition.entry("mesh/specification").binding("revision"),
                source_composition.entry("mesh/adaptation-specification").binding(
                    "revision"
                ),
                source_composition.entry("mesh/source-graph-plan").binding("revision"),
                source_composition.entry("mesh/target-graph-plan").binding("revision"),
            ),
        )
        geometry_entry = CompositionEntry(
            manifest,
            entry_id="mesh/geometry",
            role="discretization",
            owner_id="phydrax.meshing",
            structure_id=geometry_id,
            revision_id=manifest.manifest_id,
            semantics_id="compact-owner-local-geometry",
            dependencies=(
                topology_entry.binding("structure"),
                source_root_entry.binding("structure"),
                source_root_entry.binding("revision"),
                source_root_entry.binding("semantics"),
            ),
        )
        return topology_entry, geometry_entry

    def auxiliary_layouts_for(identities: Any, /) -> tuple[Any, ...]:
        return (
            phx.solver.FiniteElementRestartStateLayout(
                u_history_name,
                phx.solver.FiniteElementRestartStateRole.HISTORY,
                phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
                ownership=identities[0].ownership(),
            ),
            phx.solver.FiniteElementRestartStateLayout(
                content_history_name,
                phx.solver.FiniteElementRestartStateRole.HISTORY,
                phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
                ownership=identities[1].ownership(),
            ),
            phx.solver.FiniteElementRestartStateLayout.from_stored_fields(
                source_auxiliary_layouts[2]
            ),
        )

    adapted_entries = placement_entries(adaptation_assembly)
    target_entries = placement_entries(target_assembly)
    field_entry_ids = (
        ("u", "solver/state/field/u"),
        ("content", "solver/state/field/content"),
    )
    auxiliary_entry_ids = (
        (u_history_name, f"solver/restart/auxiliary/{u_history_name}"),
        (
            content_history_name,
            f"solver/restart/auxiliary/{content_history_name}",
        ),
        (clock_history_name, f"solver/restart/auxiliary/{clock_history_name}"),
    )
    integrator_entry_ids = (
        (previous_name, f"solver/restart/integrator/{previous_name}"),
        (rng_name, f"solver/restart/integrator/{rng_name}"),
    )
    retained_mesh_entry_ids = (
        "mesh/source",
        "mesh/specification",
        "mesh/adaptation-specification",
        "mesh/source-graph-plan",
        "mesh/target-graph-plan",
    )

    def adapt(transaction: Any) -> Any:
        return transaction.adapt_owner_local(
            restart_manifest,
            source_compiled,
            adapted_compiled,
            source_identities,
            adapted_identities,
            adaptive_result,
            source_composition,
            group,
            source_view,
            adaptation_view,
            state_entry_id="solver/state",
            geometry_entry_id="mesh/geometry",
            field_entry_ids=field_entry_ids,
            material_entry_id="solver/state/materials",
            auxiliary_entry_ids=auxiliary_entry_ids,
            integrator_entry_ids=integrator_entry_ids,
            target_auxiliary_layouts=auxiliary_layouts_for(adapted_identities),
            target_integrator_layouts=integrator_layouts,
            target_material_layouts=material_layouts,
            retain_entry_ids=retained_mesh_entry_ids,
            reprepare_entries=adapted_entries,
            axis_name="native_owner_local_fe_adaptation",
        )

    rejected = adapt(
        FiniteElementTopologyTransaction(
            lambda mesh, fields, state, lineage, context: jax.process_index() != 1,
            transaction_id="native-owner-local-fe-adaptation-reject",
        )
    )
    if (
        bool(np.asarray(rejected.committed))
        or rejected.rebind_receipt is not None
        or rejected.state.accepted_id != source_state.accepted_id
        or rejected.manifest.manifest_id != restart_manifest.manifest_id
        or source_composition.composition_id != source_composition_id
        or any(
            not np.array_equal(np.asarray(before), np.asarray(after))
            for before, after in zip(rejected.state.fields, source_fields, strict=True)
        )
    ):
        raise RuntimeError("Collective adaptation refusal changed accepted FE state.")

    def certify(mesh: Any, fields: Any, state: Any, lineage: Any, context: Any) -> bool:
        del mesh, state, lineage, context
        return bool(
            len(fields) == 2
            and np.all(np.isfinite(np.asarray(fields[0])))
            and np.all(np.asarray(fields[0]) > 0.0)
            and np.all(np.isfinite(np.asarray(fields[1])))
        )

    result = adapt(
        FiniteElementTopologyTransaction(
            certify,
            transaction_id="native-owner-local-fe-adaptation-commit",
        )
    )
    expected_stencil_id = (
        None
        if adaptive_result.vertex_stencil is None
        else adaptive_result.vertex_stencil.stencil_id
    )
    expected_lineage_id = (
        None if adaptive_result.lineage is None else adaptive_result.lineage.lineage_id
    )
    authority_checks = {
        "committed": bool(np.asarray(result.committed)),
        "published": result.rebind_receipt is not None
        and result.rebind_receipt.published
        and result.rebind_receipt.boundary_accepted,
        "topology": result.state.topology_id == adaptation_view.source_topology_id,
        "prepared": result.state.prepared_id == adapted_discretization.prepared_id,
        "compilation": result.state.compilation_id == adapted_compiled.compilation_id,
        "version": result.state.state_version == source_state.state_version + 1,
        "clock": result.state.time == source_state.time
        and result.state.step == source_state.step
        and result.state.schedule_cursor == source_state.schedule_cursor,
        "adaptive_result": result.adaptive_result_id == adaptive_result.result_id,
        "stencil": result.vertex_stencil_id == expected_stencil_id,
        "lineage": result.lineage_id == expected_lineage_id,
    }
    if not all(authority_checks.values()):
        raise RuntimeError(
            "Accepted owner-local FE topology result lost its authority: "
            f"{sorted(name for name, passed in authority_checks.items() if not passed)}; "
            f"diagnostics={result.diagnostics!r}."
        )
    transfers = {item.field_layout_id: item for item in result.field_transfers}
    u_transfer = transfers.get(source_layouts[0].layout_id)
    content_transfer = transfers.get(source_layouts[1].layout_id)
    if (
        u_transfer is None
        or u_transfer.vertex_stencil_id != expected_stencil_id
        or content_transfer is None
        or not content_transfer.conservative
        or content_transfer.conservation_evidence_id is None
        or content_transfer.conservation_bound is None
        or np.any(
            np.abs(np.asarray(content_transfer.conservation_defect))
            > content_transfer.conservation_bound
        )
    ):
        raise RuntimeError("P1 stencil/P0 conservative evidence was not committed.")

    target_auxiliary = dict(result.manifest.auxiliary_state)
    np.testing.assert_allclose(
        np.asarray(target_auxiliary[u_history_name]),
        np.asarray(result.state.fields[0]) + 2.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(target_auxiliary[content_history_name]),
        np.asarray(result.state.fields[1]) + 0.375,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(target_auxiliary[clock_history_name]), clock_history
    )
    target_integrator = dict(result.manifest.integrator_state)
    np.testing.assert_array_equal(
        np.asarray(target_integrator[previous_name]), previous_step
    )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(target_integrator[rng_name])),
        np.asarray(jax.random.key_data(rng)),
    )
    if source_state.materials is None or result.state.materials is None:
        raise RuntimeError("Topology adaptation dropped replicated material state.")
    source_material = source_state.materials.states[0]
    target_material = result.state.materials.states[0]
    if (
        source_material.site_id != target_material.site_id
        or source_material.model_id != target_material.model_id
        or source_material.state_version != target_material.state_version
    ):
        raise RuntimeError(
            "Replicated material state changed during topology adaptation."
        )
    np.testing.assert_array_equal(
        np.asarray(target_material.committed), np.asarray(source_material.committed)
    )
    np.testing.assert_array_equal(
        np.asarray(target_material.trial), np.asarray(source_material.trial)
    )
    migration = target_view.graph_migration_evidence
    if not isinstance(migration, OwnerLocalGraphMigrationEvidence):
        raise TypeError("The repartitioned FE target lacks graph migration evidence.")
    adapted_composition = result.rebind_receipt.composition
    adapted_fields = tuple(np.asarray(values) for values in result.state.fields)

    def repartition(transaction: Any) -> Any:
        return transaction.repartition(
            result.manifest,
            adapted_compiled,
            target_compiled,
            result.target_identity_plans,
            target_identities,
            migration,
            adapted_composition,
            group,
            adaptation_view,
            target_view,
            state_entry_id="solver/state",
            geometry_entry_id="mesh/geometry",
            field_entry_ids=field_entry_ids,
            material_entry_id="solver/state/materials",
            auxiliary_entry_ids=auxiliary_entry_ids,
            integrator_entry_ids=integrator_entry_ids,
            target_auxiliary_layouts=auxiliary_layouts_for(target_identities),
            target_integrator_layouts=integrator_layouts,
            target_material_layouts=material_layouts,
            retain_entry_ids=retained_mesh_entry_ids,
            reprepare_entries=target_entries,
            axis_name="native_owner_local_fe_adapted_repartition",
        )

    moved = repartition(
        FiniteElementTopologyTransaction(
            lambda mesh, fields, state, lineage, context: True,
            transaction_id="native-owner-local-fe-adapted-repartition",
        )
    )
    if (
        not bool(np.asarray(moved.committed))
        or moved.rebind_receipt is None
        or not moved.rebind_receipt.published
        or moved.state.topology_id != target_view.source_topology_id
        or moved.state.prepared_id != target_discretization.prepared_id
        or moved.state.compilation_id != target_compiled.compilation_id
        or moved.state.time != result.state.time
        or moved.state.step != result.state.step
    ):
        raise RuntimeError(
            f"Adapted FE state did not repartition onto its target: {moved.diagnostics}"
        )
    moved_auxiliary = dict(moved.manifest.auxiliary_state)
    np.testing.assert_allclose(
        np.asarray(moved_auxiliary[u_history_name]),
        np.asarray(moved.state.fields[0]) + 2.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(moved_auxiliary[content_history_name]),
        np.asarray(moved.state.fields[1]) + 0.375,
        rtol=0.0,
        atol=2.0e-12,
    )
    moved_integrator = dict(moved.manifest.integrator_state)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(moved_integrator[rng_name])),
        np.asarray(jax.random.key_data(rng)),
    )

    next_solve = _solve_physical_problem(
        target_compiled,
        tuple(
            int(dof_map.global_dof_count) for dof_map in target_discretization.dof_maps
        ),
    )
    if (
        not next_solve["successful"]
        or not np.all(np.isfinite(np.asarray(next_solve["values"])))
        or not np.all(np.isfinite(np.asarray(next_solve["content_values"])))
    ):
        raise RuntimeError(
            "The accepted target FE state is not followed by a valid solve."
        )

    rows = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    int(np.asarray(adaptation_view.owned_cell_ids).size),
                    int(adapted_fields[0].size),
                    int(np.asarray(target_view.owned_cell_ids).size),
                    int(np.asarray(moved.state.fields[0]).size),
                    int(np.asarray(result.committed)),
                    int(np.asarray(moved.committed)),
                    int(bool(next_solve["successful"])),
                ),
                dtype=np.int64,
            )
        ),
        dtype=np.int64,
    )
    nonconstant = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    float(np.min(np.asarray(source_fields[1]), initial=np.inf)),
                    float(np.max(np.asarray(source_fields[1]), initial=-np.inf)),
                ),
                dtype=np.float64,
            )
        ),
        dtype=np.float64,
    )
    if not float(np.max(nonconstant[:, 1]) - np.min(nonconstant[:, 0])) > 1.0e-5:
        raise RuntimeError("The conservative P0 test field is not genuinely nonconstant.")

    # Independent oracle: the owned content integral (P0 average times cell measure,
    # summed over every owner) of the nonconstant field is preserved exactly up to
    # rounding of the absolute content, through the adaptation and the repartition.
    def global_content(
        view: Any, discretization: Any, identity: Any, values: Any
    ) -> float:
        return float(
            np.sum(
                multihost_utils.process_allgather(
                    np.asarray(
                        _periodic_p0_owner_integral(
                            view, discretization, identity, values
                        ),
                        dtype=np.float64,
                    )
                )
            )
        )

    source_content = global_content(
        source_view, source_discretization, source_identities[1], source_fields[1]
    )
    source_absolute = global_content(
        source_view,
        source_discretization,
        source_identities[1],
        np.abs(np.asarray(source_fields[1], dtype=np.float64)),
    )
    bound = 64.0 * np.finfo(np.float64).eps * max(1.0, source_absolute)
    adapted_content = global_content(
        adaptation_view,
        adapted_discretization,
        result.target_identity_plans[1],
        adapted_fields[1],
    )
    target_content = global_content(
        target_view,
        target_discretization,
        moved.target_identity_plans[1],
        moved.state.fields[1],
    )
    if (
        not source_absolute > 0.0
        or abs(adapted_content - source_content) > bound
        or abs(target_content - source_content) > bound
    ):
        raise RuntimeError(
            "Owner-local P0 adaptation/repartition changed the global nonconstant "
            f"content: {source_content!r} -> {adapted_content!r} -> {target_content!r}."
        )
    print(
        json.dumps(
            {
                "phase": "adapted",
                "rank": rank,
                "rows": rows.tolist(),
                "p0_conservation_defect": np.asarray(
                    content_transfer.conservation_defect
                ).tolist(),
            },
            sort_keys=True,
        )
    )
