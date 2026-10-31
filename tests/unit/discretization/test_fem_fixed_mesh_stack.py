#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mesh() -> Any:
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(
        vertices, cells, cell_global_ids=jnp.asarray([10, 20])
    )


def test_fem_fixed_mesh_stack_scenario_1() -> None:
    mesh = _mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.discontinuous_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.fem.sipg_poisson_form(
        "u",
        1.0,
        phx.equations.fem.SIPGPenaltyPolicy(12.0),
        discretization.cell_domain,
        discretization.interior_facet_domain,
        (),
    )
    action_ir = phx.equations.fem.lower_finite_element_form(form, discretization)
    workset_program = phx.equations.fem.compile_workset_program(
        action_ir, form, discretization
    )
    facet = tuple(
        workset
        for workset in workset_program.worksets
        if workset.signature.region_kind == "interior-facet"
    )[0]

    assert facet.action_indices.shape == (1,)
    assert facet.owner_local_entities.shape == facet.entity_indices.shape
    assert facet.neighbor_local_entities.shape == facet.entity_indices.shape
    assert int(facet.owner_permutations[0]) == -int(facet.neighbor_permutations[0])
    assert dict(facet.neighbor_gathers)["u"].shape == dict(facet.gathers)["u"].shape
    matrices = jnp.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    inputs = jnp.asarray([[0, 1]], dtype=jnp.int32)
    outputs = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    operator = phx.sparse.ElementTensorOperator(matrices, inputs, outputs, 2, 3)
    value = jnp.asarray([2.0, -1.0])
    covector = jnp.asarray([1.0, 2.0, -1.0])

    assert jnp.allclose(operator.mv(value), matrices[0] @ value)
    assert jnp.allclose(operator.transpose_mv(covector), matrices[0].T @ covector)
    assert jnp.allclose(operator.as_sparse_coordinate().mv(value), operator.mv(value))
    mesh = _mesh()
    element = phx.discretization.lagrange_element("triangle", 3)
    discretization = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", element)
    ).prepare()
    projected = discretization.project(
        "u", lambda points, args: points[:, 0] ** 2 + points[:, 1] ** 2
    )

    family = phx.discretization.fem.ReferenceNodalFamily("quadrilateral", 1)
    tabulation = phx.discretization.fem.TensorProductTabulation(
        family, (jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0]))
    )
    plan = phx.discretization.fem.SumFactorizationPlan(tabulation)
    tensor = phx.equations.fem.TensorProductPartialAssemblyOperator(
        plan,
        jnp.ones((1, 2, 2)),
        jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
        4,
        action_kind="mass",
    )

    assert discretization.dof_maps[0].association == "entity"
    assert projected.shape == (16,)
    assert jnp.allclose(tensor.mv(jnp.arange(1.0, 5.0)), jnp.arange(1.0, 5.0))


def test_block_graph_preconditioner_data_and_auxiliary_validity() -> None:
    mesh = _mesh()
    p1 = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec(
                "u",
                phx.discretization.lagrange_element("triangle", 2),
                component_shape=(2,),
            ),
            phx.discretization.FiniteElementFieldSpec("p", p1),
        ),
    ).prepare()
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.fem.stokes_form("u", "p"), discretization
    )
    zero = compiled.state_space.zeros()
    block = compiled.block_linearization_operator(zero)

    assert compiled.block_dependency_graph() == ((True, True), (True, False))
    assert tuple(value.shape for value in block.mv(zero)) == ((9, 2), (4,))

    scalar = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("s", p1)
    ).prepare()

    def auxiliary(state: Any, context: Any) -> Any:
        return phx.equations.fem.FiniteElementAuxiliaryEvaluation(
            state + 1.0, successful=True, admissible=True
        )

    form = phx.equations.FiniteElementForm(
        "auxiliary",
        "s",
        (phx.equations.DiffusionAction("s"),),
        auxiliary_evaluator=auxiliary,
        auxiliary_id="state-plus-one",
    )
    scalar_compiled = phx.equations.compile_finite_element_problem(form, scalar)
    _, evaluated = scalar_compiled.residual_with_auxiliary(jnp.zeros((4,)))
    data = scalar_compiled.preconditioner_data()

    assert evaluated.valid
    # ty: ignore[invalid-argument-type]
    assert jnp.allclose(evaluated.trial_state, 1.0)
    # ty: ignore[unresolved-attribute]
    assert data.diagonal.shape == (4,)


def test_observation_restart_and_result_roundtrip(tmp_path: Any) -> None:
    space = phx.linalg.ArraySpace((4,))
    observation = phx.equations.fem.CoordinateObservation(
        space, jnp.asarray([1, 3]), weights=jnp.asarray([2.0, 1.0])
    )
    state = jnp.arange(4.0)
    test_value = jnp.asarray([3.0, 4.0])
    assert jnp.allclose(
        jnp.vdot(observation.evaluate(state), test_value),
        jnp.vdot(state, observation.transpose(test_value)),
    )

    layout = phx.solver.FiniteElementAcceptedFieldLayout(
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (state,),
        0.5,
        3,
        "topology",
        "prepared",
        "compiled",
        field_layouts=(layout,),
    )
    manifest = phx.solver.FiniteElementRestartManifest(
        accepted,
        auxiliary_state=(("history", jnp.asarray([2.0])),),
        integrator_state=(("previous", jnp.asarray([0.25])),),
    )
    restart_path = tmp_path / "restart.npz"
    phx.solver.write_finite_element_restart(restart_path, manifest)
    restored = phx.solver.read_finite_element_restart(
        restart_path,
        field_layouts=(layout,),
    )
    assert restored.manifest_id == manifest.manifest_id

    diagnostics = phx.solver.FiniteElementSolveDiagnostics(True, 1.0e-12)
    result = phx.solver.FiniteElementResult(
        ("u",), (state,), 0.5, "prepared", "compiled", diagnostics
    )
    result_path = tmp_path / "result.npz"
    phx.solver.write_finite_element_result(result_path, result)
    loaded = phx.solver.read_finite_element_result(result_path)
    assert loaded.result_id == result.result_id


def test_contact_search_and_cut_quadrature_are_deterministic() -> None:
    collision = phx.discretization.contact
    source = phx.linalg.ArraySpace((6, 2), dtype=np.float64)
    slave_plan = collision.CollisionSurfacePlan(
        jnp.asarray([3, 4]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1]], dtype=jnp.int32),
    )
    master_plan = collision.CollisionSurfacePlan(
        jnp.asarray([5, 6, 7, 8]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        body_ids=1,
        material_ids=0,
    )
    slave = collision.PreparedCollisionSurface(
        slave_plan,
        jnp.asarray([[0.5, 0.1], [0.6, 0.1]]),
        collision.selection_collision_operator(source, jnp.asarray([0, 1])),
    )
    master = collision.PreparedCollisionSurface(
        master_plan,
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        collision.selection_collision_operator(source, jnp.asarray([2, 3, 4, 5])),
    )
    scene = collision.PreparedCollisionScene((slave, master))
    search = collision.DenseContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.5,
    )
    epoch = search.build(scene, scene.positions(source.zeros()))
    repeated = search.build(scene, scene.positions(source.zeros()))
    np.testing.assert_array_equal(
        epoch.edge_vertex.route_keys, repeated.edge_vertex.route_keys
    )

    mesh = _mesh()
    fracture = phx.applications.fracture
    crack = fracture.CrackFrontGeometry(
        jnp.asarray([[0.0, 0.5], [1.0, 0.5]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
    )
    topology = fracture.build_sharp_crack_topology(mesh, crack)
    quadrature = fracture.build_sharp_crack_quadrature(mesh, topology)

    integrated_area = (
        jnp.sum(quadrature.plus.weights)
        + jnp.sum(quadrature.minus.weights)
        + jnp.sum(quadrature.tips.weights)
    )
    assert jnp.allclose(integrated_area, 1.0)
    assert topology.heaviside_vertex_ids.size == mesh.vertex_global_ids.size


def test_compiled_fe_owner_local_repartition_rebinds_accepted_restart() -> None:
    import jax
    import pytest

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._partition import CellPartition
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.discretization.fem._distributed import finite_element_dof_identity_plan
    from phydrax.graph import (
        DistributedWeightedCSRGraph,
        GraphPartitionPlan,
        partition_graph_distributed,
    )
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._assembly import MeshPart, MeshPartManifest
    from phydrax.meshing._distribution import (
        MeshDistribution,
        OwnerLocalGraphMigrationEvidence,
        ShardedOwnerDirectoryWitness,
    )
    from phydrax.meshing._result import CellMeshingResult

    if jax.process_count() != 1:
        pytest.skip("This FE repartition contract uses one local process group.")

    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "compiled-fe-owner-repartition",
            (process,),
            ((process, device.id),),
            mesh_axes=(("fe_repartition", 1),),
        ),
        (device,),
    )

    part = MeshPart(
        "fe-owner-repartition",
        phx.meshing.certify_cell_mesh(_mesh(), phx.SpatialCoordinateContract.si()),
    )
    if not isinstance(part.carrier, CellMeshingResult):
        raise TypeError("The one-rank FE fixture requires a complete cell mesh.")
    mesh = part.carrier.mesh
    geometry = part.carrier.geometry
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("The one-rank FE fixture requires an exact cell geometry spec.")
    source_mesh_manifest = MeshPartManifest.from_part(part)
    if source_mesh_manifest.source_geometry_id is None:
        raise ValueError("The certified source mesh lacks global geometry identity.")
    global_geometry_id = source_mesh_manifest.source_geometry_id
    geometry_id = cell_geometry_id(geometry)
    if geometry_id != global_geometry_id:
        raise ValueError("The exact coordinate spec differs from its global manifest.")
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    cell_owners = np.zeros(cell_ids.shape, dtype=np.int32)
    distribution = MeshDistribution(part, CellPartition(cell_owners, 1), halo_width=0)
    local = distribution.local_cell_geometry(0, include_halo=False)
    local_cell_ids = np.asarray(local["cell_ids"], dtype=np.int64)
    local_vertex_ids = np.asarray(local["vertex_ids"], dtype=np.int64)
    source_point_ids = np.asarray(local["coordinate_rows"], dtype=np.int64)
    zero_owners = np.zeros(local_cell_ids.shape, dtype=np.int32)
    zero_vertex_owners = np.zeros(local_vertex_ids.shape, dtype=np.int32)
    zero_source_owners = np.zeros(source_point_ids.shape, dtype=np.int32)
    base_directory = ShardedOwnerDirectoryWitness(
        group,
        0,
        source_mesh_manifest.manifest_id,
        "mesh/owner-directory",
        source_point_ids,
        zero_source_owners,
        np.ones(source_point_ids.shape, dtype=np.bool_),
        local_cell_ids,
        zero_owners,
        local_vertex_ids,
        zero_vertex_owners,
        source_map_id=geometry.source_map_id,
        axis_name="compiled_fe_repartition_directory",
    )
    source_entry = CompositionEntry(
        part.carrier,
        entry_id="mesh/source",
        role="topology",
        owner_id="test-discretization-source",
        structure_id=mesh.mesh_id,
        revision_id=part.part_id,
        semantics_id=geometry_id,
    )
    part_authority = CompositionEntry(
        part,
        entry_id="mesh/part-authority",
        role="topology",
        owner_id="test-discretization",
        structure_id=mesh.topology_id,
        revision_id=part.part_id,
        semantics_id=geometry_id,
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
            base_directory.composition_entry().binding("structure"),
        ),
    )
    owner_directory = base_directory.bind_source_authority(
        part_authority.record_id,
        group,
        axis_name="compiled_fe_repartition_directory",
    )
    accepted_geometry = Composition(
        (source_entry, owner_directory.composition_entry(), part_authority),
        boundary_id=source_mesh_manifest.manifest_id,
    )
    owner_record = distribution.owner_local_geometry_record(
        0,
        part_authority,
        accepted_geometry,
        owner_directory,
        source_id=part.name,
        source_revision=part.part_id,
        include_halo=False,
    )
    source_geometry_view = OwnerLocalCellGeometry.from_owner_local_result(owner_record)
    target_geometry_view = source_geometry_view
    geometry_part = MeshPart(
        part.name,
        source_geometry_view,
        coordinate_contract=part.coordinate_contract,
    )
    mesh_manifest = MeshPartManifest.from_part(geometry_part)
    source_directory_id = source_geometry_view.distribution_id
    target_directory_id = source_geometry_view.distribution_id
    cell_ids = np.asarray(source_geometry_view.cell_ids, dtype=np.int64)
    cell_owners = np.asarray(source_geometry_view.cell_owner_ranks, dtype=np.int32)
    vertex_ids = np.asarray(source_geometry_view.vertex_ids, dtype=np.int64)
    vertex_owners = np.asarray(source_geometry_view.vertex_owner_ranks, dtype=np.int32)
    field_spec = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    field_layout = phx.solver.FiniteElementAcceptedFieldLayout(
        field_spec,
        transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    source_discretization = phx.discretization.FiniteElementPlan(
        mesh, field_spec, coordinate_spec=geometry
    ).prepare()
    target_discretization = phx.discretization.FiniteElementPlan(
        mesh, field_spec, coordinate_spec=geometry
    ).prepare()

    def compile_problem(discretization: Any) -> Any:
        form = phx.equations.FiniteElementForm(
            "owner-local-h1-diffusion-source",
            "u",
            (
                phx.equations.DiffusionAction("u"),
                phx.equations.SourceAction("u", 1.0),
            ),
        )
        return phx.equations.compile_finite_element_problem(form, discretization)

    source_compiled = compile_problem(source_discretization)
    target_compiled = compile_problem(target_discretization)
    source_identity = finite_element_dof_identity_plan(
        source_discretization,
        0,
        field_layout_id=field_layout.layout_id,
        distribution_id=source_directory_id,
        topology_id=mesh.topology_id,
        geometry_id=geometry_id,
        local_rank=process,
        part_count=1,
        cell_ids=cell_ids,
        cell_owner_ranks=cell_owners,
        vertex_ids=vertex_ids,
        vertex_owner_ranks=vertex_owners,
    )
    target_identity = finite_element_dof_identity_plan(
        target_discretization,
        0,
        field_layout_id=field_layout.layout_id,
        distribution_id=target_directory_id,
        topology_id=mesh.topology_id,
        geometry_id=geometry_id,
        local_rank=process,
        part_count=1,
        cell_ids=cell_ids,
        cell_owner_ranks=cell_owners,
        vertex_ids=vertex_ids,
        vertex_owner_ranks=vertex_owners,
    )
    field_values = np.linspace(
        0.25,
        1.25,
        source_discretization.dof_maps[0].global_dof_count,
        dtype=np.float64,
    )
    source_values, source_ownership = source_identity.shard_values(field_values)
    state = phx.solver.FiniteElementAcceptedState(
        (source_values,),
        np.asarray(0.125, dtype=np.float64),
        4,
        mesh.topology_id,
        source_discretization.prepared_id,
        source_compiled.compilation_id,
        field_layouts=(field_layout,),
        field_ownership=(source_ownership,),
    )
    history_layout = phx.solver.FiniteElementRestartStateLayout(
        "accepted-history",
        phx.solver.FiniteElementRestartStateRole.HISTORY,
        phx.solver.FiniteElementRestartStateAssociation.REPLICATED,
    )
    history_value = np.asarray((2.0, 3.0), dtype=np.float64)
    source_manifest = phx.solver.FiniteElementRestartManifest(
        state,
        auxiliary_state=(("accepted-history", history_value),),
        auxiliary_layouts=(history_layout,),
    )
    source_root_entry = CompositionEntry(
        part,
        entry_id="mesh/source-root",
        role="topology",
        owner_id="test-discretization-source",
        structure_id=source_geometry_view.source_mesh_id,
        revision_id=source_geometry_view.source_revision,
        semantics_id=source_geometry_view.source_geometry_id,
    )
    mesh_entry = CompositionEntry(
        mesh_manifest,
        entry_id="mesh/topology",
        role="topology",
        owner_id="test-discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh_manifest.manifest_id,
        semantics_id="cell-mesh",
    )
    geometry_entry = CompositionEntry(
        mesh_manifest,
        entry_id="mesh/geometry",
        role="discretization",
        owner_id="test-discretization",
        structure_id=global_geometry_id,
        revision_id=mesh_manifest.manifest_id,
        semantics_id="cell-geometry-manifest",
        dependencies=(
            mesh_entry.binding("structure"),
            source_root_entry.binding("structure"),
            source_root_entry.binding("revision"),
            source_root_entry.binding("semantics"),
        ),
    )
    state_entry = CompositionEntry(
        state,
        entry_id="solver/state",
        role="physical-state",
        owner_id="test-finite-element-solver",
        structure_id=mesh.topology_id,
        revision_id=state.accepted_id,
        semantics_id="accepted-fe-state",
        dependencies=(
            mesh_entry.binding("structure"),
            geometry_entry.binding("structure"),
        ),
    )
    field_entry = CompositionEntry(
        source_values,
        entry_id="solver/state/field/u",
        role="physical-state",
        owner_id=state_entry.owner_id,
        structure_id=mesh.topology_id,
        revision_id=state.accepted_id,
        semantics_id=field_layout.layout_id,
        dependencies=(state_entry.binding("revision"),),
    )
    history_entry = CompositionEntry(
        history_value,
        entry_id="solver/restart/auxiliary/accepted-history",
        role="history",
        owner_id=state_entry.owner_id,
        structure_id=history_layout.layout_id,
        revision_id=source_manifest.manifest_id,
        semantics_id=history_layout.semantics_id,
        dependencies=(state_entry.binding("structure"),),
    )
    source_composition = Composition(
        (
            source_root_entry,
            mesh_entry,
            geometry_entry,
            state_entry,
            field_entry,
            history_entry,
        ),
        boundary_id="source-fe-accepted",
    )

    graph = DistributedWeightedCSRGraph(
        np.zeros((cell_ids.size + 1,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        entity_ids=cell_ids,
        owner_ranks=cell_owners,
        neighbor_owner_ranks=np.empty((0,), dtype=np.int32),
        vertex_weights=np.ones(cell_ids.shape, dtype=np.int64),
    )
    graph_result = partition_graph_distributed(
        graph,
        GraphPartitionPlan(1, maximum_imbalance=1.0),
        group,
        axis_name="compiled_fe_repartition_partition",
    )
    migration = OwnerLocalGraphMigrationEvidence(
        graph_result,
        source_mesh_manifest.manifest_id,
        source_geometry_view.part_id,
        source_geometry_view.source_geometry_id,
        source_directory_id,
        target_geometry_view.part_id,
        target_directory_id,
        group.spec.group_id,
        process,
        1,
        cell_ids,
    )
    result = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate, fields, materials, lineage, args: True
    ).repartition(
        source_manifest,
        source_compiled,
        target_compiled,
        (source_identity,),
        (target_identity,),
        migration,
        source_composition,
        group,
        source_geometry_view,
        target_geometry_view,
        state_entry_id="solver/state",
        geometry_entry_id=geometry_entry.entry_id,
        field_entry_ids=(("u", field_entry.entry_id),),
        material_entry_id=None,
        auxiliary_entry_ids=(("accepted-history", history_entry.entry_id),),
        target_auxiliary_layouts=(history_layout,),
        retain_entry_ids=(mesh_entry.entry_id, source_root_entry.entry_id),
        reprepare_entries=(geometry_entry,),
    )
    assert bool(result.committed), f"FE repartition refusal: {result.diagnostics!r}"
    assert result.rebind_receipt is not None
    assert result.rebind_receipt.published
    assert result.state.prepared_id == target_discretization.prepared_id
    assert result.state.compilation_id == target_compiled.compilation_id
    np.testing.assert_array_equal(
        np.asarray(result.state.fields[0]).view(np.uint8),
        np.asarray(source_values).view(np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(result.manifest.auxiliary_state[0][1]).view(np.uint8),
        history_value.view(np.uint8),
    )
    target_composition = result.rebind_receipt.composition
    assert (
        target_composition.entry(field_entry.entry_id).revision_id
        == result.state.accepted_id
    )
    assert (
        target_composition.entry(history_entry.entry_id).revision_id
        == result.manifest.manifest_id
    )
    residual = target_compiled.full_residual(result.state.fields[0])
    assert all(
        np.all(np.isfinite(np.asarray(value))) for value in jax.tree.leaves(residual)
    )
