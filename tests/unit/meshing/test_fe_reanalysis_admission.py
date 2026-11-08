from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.fem._reference import FiniteElementSpec
from tools.meshing_qualification.scenarios.native_design import (
    _physical_solve,
    _sphere_source,
)


def test_surface_content_refuses_custom_nominal_p1_tabulator() -> None:
    base = phx.discretization.lagrange_element("triangle", 1)

    def altered(points: Any) -> tuple[Any, Any]:
        values, gradients = base.tabulate(points)
        u, v = points[:, 0], points[:, 1]
        bubble = u * v
        values = values.at[:, 0].add(bubble).at[:, 1].add(-bubble)
        derivative = jnp.stack((v, u), axis=1)
        gradients = gradients.at[:, 0, :].add(derivative).at[:, 1, :].add(-derivative)
        return values, gradients

    custom = FiniteElementSpec(
        base.family,
        base.cell_kind,
        base.degree,
        base.reference_nodes,
        base.entity_dofs,
        value_spec=base.value_spec,
        continuity=base.continuity,
        representation=base.representation,
        tabulator=altered,
        tabulator_id="noncanonical-p1",
    )
    field = phx.discretization.FiniteElementFieldSpec("density", custom)
    with pytest.raises(ValueError, match="canonical"):
        phx.solver.FiniteElementAcceptedFieldLayout(
            field,
            transfer_semantics=phx.solver.FiniteElementFieldTransferSemantics.CONSERVATIVE_SURFACE_CONTENT,
        )


@pytest.fixture(scope="module")
def case() -> tuple[Any, ...]:
    _, _, _, _, source, _ = _sphere_source(2)
    source_solve = _physical_solve(source, 0.75)
    scope = phx.meshing.mesh_proposal_scope(source, 2)
    proposal = phx.meshing.MeshMarkingProposal(
        source, scope, np.ones(scope.entity_ids.size), proposer_id="reanalysis-admission"
    )
    safety = phx.meshing.MeshProposalSafetyPolicy(
        source, minimum_size=1.0e-6, maximum_size=2.8, maximum_displacement=0.0
    )
    tx = phx.meshing.prepare_mesh_proposal(
        source,
        proposal,
        safety,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    adaptation = tx.adaptation
    assert adaptation is not None
    target = tx.commit(source)
    target_solve = _physical_solve(target, 0.75)
    element = phx.discretization.lagrange_element("triangle", 1)
    fields = tuple(
        phx.discretization.FiniteElementFieldSpec(name, element)
        for name in ("u", "density")
    )
    semantics = phx.solver.FiniteElementFieldTransferSemantics
    layouts = (
        phx.solver.FiniteElementAcceptedFieldLayout(
            fields[0], transfer_semantics=semantics.NONCONSERVATIVE_INTERPOLATION
        ),
        phx.solver.FiniteElementAcceptedFieldLayout(
            fields[1], transfer_semantics=semantics.CONSERVATIVE_SURFACE_CONTENT
        ),
    )
    source_all = phx.discretization.FiniteElementPlan(
        source.mesh, fields, coordinate_spec=source.geometry
    ).prepare()
    target_all = phx.discretization.FiniteElementPlan(
        target.mesh, fields, coordinate_spec=target.geometry
    ).prepare()
    full_form = phx.equations.FiniteElementForm(
        "reanalysis-full-state", ("u", "density"), source_solve["problem"].form.actions
    )
    source_program = phx.equations.compile_finite_element_problem(full_form, source_all)
    target_program = phx.equations.compile_finite_element_problem(full_form, target_all)
    accepted = phx.solver.FiniteElementAcceptedState(
        (source_solve["values"], jnp.full(source.mesh.coordinates.shape[0], 2.0)),
        0.0,
        0,
        source.mesh.topology_id,
        source_all.prepared_id,
        source_program.compilation_id,
        field_layouts=layouts,
    )
    return source, adaptation, target, target_solve, accepted, target_all, target_program


def _executor() -> phx.solver.FiniteElementTopologyTransaction:
    return phx.solver.FiniteElementTopologyTransaction(
        lambda mesh, fields, materials, lineage, args: True
    )


def test_reanalysis_does_not_borrow_threshold_from_different_rhs(
    case: tuple[Any, ...],
) -> None:
    source, adaptation, target, _, accepted, _, _ = case
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        target.mesh, field, coordinate_spec=target.geometry
    ).prepare()

    def scaled(points: Any, args: Any) -> Any:
        scale = 1.0 if args.user_args is None else args.user_args
        return points[..., 0] ** 2 * scale

    form = phx.equations.FiniteElementForm(
        "scaled-reanalysis",
        "u",
        (
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction(
                "u", phx.equations.coefficient(scaled, coefficient_id="scaled-load")
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, space)
    system, rhs = problem.linear_system(1.0e12)
    large, check = phx.linalg.solve_checked(system, rhs)
    bad = eqx.tree_at(lambda value: value.value, large, jnp.zeros_like(large.value))
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        problem, bad, check, "u", check_policy=phx.linalg.LinearSolveCheckPolicy()
    )
    result = _executor().execute(
        accepted, source.mesh, adaptation, 1.0, reanalysis_action=lambda *args: reanalysis
    )
    assert not bool(np.asarray(result.committed))
    assert result.state is accepted
    assert result.reanalysis is not None
    assert not bool(np.asarray(result.reanalysis.solve_check.valid))


def test_reanalysis_cannot_replace_conservative_density_with_stale_proof(
    case: tuple[Any, ...],
) -> None:
    source, adaptation, target, _, accepted, _, _ = case
    field = phx.discretization.FiniteElementFieldSpec(
        "density", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        target.mesh, field, coordinate_spec=target.geometry
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "different-density",
        "density",
        (
            phx.equations.MassAction("density", 1.0),
            phx.equations.SourceAction(
                "density",
                phx.equations.coefficient(
                    lambda points, args: jnp.full(points.shape[:-1], 3.0),
                    coefficient_id="density-three",
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, space)
    system, rhs = problem.linear_system()
    solved, check = phx.linalg.solve_checked(system, rhs)
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        problem,
        solved,
        check,
        "density",
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    result = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: reanalysis
    )
    assert not bool(np.asarray(result.committed))
    assert result.state is accepted


def test_one_field_reanalysis_keeps_complete_state_program_ids(
    case: tuple[Any, ...],
) -> None:
    source, adaptation, _, target_solve, accepted, target_all, target_program = case
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=target_program,
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    result = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: reanalysis
    )
    assert bool(np.asarray(result.committed))
    assert result.state.prepared_id == target_all.prepared_id
    assert result.state.compilation_id == target_program.compilation_id
    assert target_program.form.field_names == tuple(
        layout.name for layout in result.state.field_layouts
    )
    incomplete = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=target_solve["problem"],
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    refused = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: incomplete
    )
    assert not bool(np.asarray(refused.committed))
    assert refused.state is accepted


def test_reanalysis_refuses_wrong_nonrefreshed_component_shape(
    case: tuple[Any, ...],
) -> None:
    source, adaptation, target, target_solve, accepted, _, _ = case
    element = phx.discretization.lagrange_element("triangle", 1)
    fields = (
        phx.discretization.FiniteElementFieldSpec("u", element),
        phx.discretization.FiniteElementFieldSpec(
            "density", element, component_shape=(2,)
        ),
    )
    space = phx.discretization.FiniteElementPlan(
        target.mesh, fields, coordinate_spec=target.geometry
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "wrong-density-fiber", ("u", "density"), target_solve["problem"].form.actions
    )
    program = phx.equations.compile_finite_element_problem(form, space)
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=program,
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    result = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: reanalysis
    )
    assert not bool(np.asarray(result.committed))
    assert result.state is accepted


def test_reanalysis_refuses_wrong_complete_state_dtype(case: tuple[Any, ...]) -> None:
    source, adaptation, target, target_solve, accepted, _, _ = case
    fields = tuple(layout.field_spec for layout in accepted.field_layouts)
    space = phx.discretization.FiniteElementPlan(
        target.mesh,
        fields,
        coordinate_spec=target.geometry,
        coefficient_dtype=jnp.float32,
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "wrong-state-dtype", ("u", "density"), target_solve["problem"].form.actions
    )
    program = phx.equations.compile_finite_element_problem(form, space)
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=program,
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    result = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: reanalysis
    )
    assert not bool(np.asarray(result.committed))
    assert result.state is accepted


def test_reanalysis_accepts_other_fields_actual_actions(case: tuple[Any, ...]) -> None:
    source, adaptation, _, target_solve, accepted, target_all, _ = case
    actions = target_solve["problem"].form.actions + (
        phx.equations.MassAction("density", 1.0, action_id="density-mass"),
        phx.equations.SourceAction("density", 2.0, action_id="density-source"),
    )
    form = phx.equations.FiniteElementForm(
        "complete-state-actions", ("u", "density"), actions
    )
    program = phx.equations.compile_finite_element_problem(form, target_all)
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=program,
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    result = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: reanalysis
    )
    assert bool(np.asarray(result.committed))
    assert result.state.compilation_id == program.compilation_id


def test_reanalysis_preserves_explicit_nullspace_requirement(
    case: tuple[Any, ...],
) -> None:
    source, adaptation, _, target_solve, accepted, _, target_program = case
    required = phx.linalg.LinearSolveCheckPolicy(require_nullspace=True)
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=target_program,
        check_policy=required,
    )
    result = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: reanalysis
    )
    assert not bool(np.asarray(result.committed))
    assert result.state is accepted


def test_surface_content_refuses_complex_payload(case: tuple[Any, ...]) -> None:
    source, adaptation, _, _, accepted, _, _ = case
    complex_density = accepted.fields[1].astype(jnp.complex128) + 1j
    changed = phx.solver.FiniteElementAcceptedState(
        (accepted.fields[0], complex_density),
        accepted.time,
        accepted.step,
        accepted.topology_id,
        accepted.prepared_id,
        accepted.compilation_id,
        field_layouts=accepted.field_layouts,
    )
    result = _executor().execute(changed, source.mesh, adaptation)
    assert not bool(np.asarray(result.committed))
    assert result.state is changed


def test_reanalysis_operator_identity_binds_current_runtime_coefficient(
    case: tuple[Any, ...],
) -> None:
    source, adaptation, target, _, accepted, target_all, _ = case
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    scalar_space = phx.discretization.FiniteElementPlan(
        target.mesh, field, coordinate_spec=target.geometry
    ).prepare()

    def mass_scale(points: Any, context: Any) -> Any:
        factor = 1.0 if context.user_args is None else context.user_args
        return jnp.full(points.shape[:-1], factor)

    actions = (
        phx.equations.MassAction(
            "u", phx.equations.coefficient(mass_scale, coefficient_id="runtime-mass")
        ),
        phx.equations.SourceAction("u", 1.0),
    )
    scalar_form = phx.equations.FiniteElementForm("runtime-mass-test", "u", actions)
    full_form = phx.equations.FiniteElementForm(
        "runtime-mass-test", ("u", "density"), actions
    )
    scalar = phx.equations.compile_finite_element_problem(scalar_form, scalar_space)
    state_program = phx.equations.compile_finite_element_problem(full_form, target_all)
    identities = []
    rhs_values = []
    for factor in (1.0, 2.0):
        system, rhs = scalar.linear_system(factor)
        solution, check = phx.linalg.solve_checked(system, rhs)
        record = phx.solver.FiniteElementReanalysisResult(
            scalar,
            solution,
            check,
            "u",
            state_problem=state_program,
            check_policy=phx.linalg.LinearSolveCheckPolicy(),
        )
        result = _executor().execute(
            accepted,
            source.mesh,
            adaptation,
            factor,
            reanalysis_action=lambda *args: record,
        )
        assert bool(np.asarray(result.committed))
        assert result.reanalysis is not None
        identities.append(result.reanalysis.operator_numeric_id)
        rhs_values.append(np.asarray(rhs))
    np.testing.assert_array_equal(rhs_values[0], rhs_values[1])
    assert identities[0] != identities[1]


def test_complete_state_program_is_real_repartition_authority(
    case: tuple[Any, ...],
) -> None:
    import jax

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
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
    from phydrax.solver._finite_element_adaptivity import (
        _FiniteElementRepartitionContext,
        _validate_finite_element_repartition_authority,
    )

    if jax.process_count() != 1:
        pytest.skip("This authority test uses one real owner rank.")
    source, adaptation, target, target_solve, accepted, target_all, target_program = case
    reanalysis = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=target_program,
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    promoted = _executor().execute(
        accepted, source.mesh, adaptation, reanalysis_action=lambda *args: reanalysis
    )
    assert bool(np.asarray(promoted.committed))
    state = promoted.state
    assert state.prepared_id == target_all.prepared_id
    assert state.compilation_id == target_program.compilation_id

    rank = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == rank)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "reanalysis-repartition",
            (rank,),
            ((rank, device.id),),
            mesh_axes=(("reanalysis_repartition", 1),),
        ),
        (device,),
    )
    part = MeshPart("reanalysis-target", target)
    manifest = MeshPartManifest.from_part(part)
    geometry_id = cell_geometry_id(target.geometry)
    assert manifest.source_geometry_id == geometry_id
    cell_ids = np.asarray(target.mesh.blocks[0].global_ids, dtype=np.int64)
    distribution = MeshDistribution(
        part, CellPartition(np.zeros(cell_ids.shape, dtype=np.int32), 1), halo_width=0
    )
    local = distribution.local_cell_geometry(0, include_halo=False)
    local_cells = np.asarray(local["cell_ids"], dtype=np.int64)
    vertices = np.asarray(local["vertex_ids"], dtype=np.int64)
    points = np.asarray(local["coordinate_rows"], dtype=np.int64)
    cell_owners = np.zeros(local_cells.shape, dtype=np.int32)
    vertex_owners = np.zeros(vertices.shape, dtype=np.int32)
    directory = ShardedOwnerDirectoryWitness(
        group,
        0,
        manifest.manifest_id,
        "mesh/owner-directory",
        points,
        np.zeros(points.shape, dtype=np.int32),
        np.ones(points.shape, dtype=np.bool_),
        local_cells,
        cell_owners,
        vertices,
        vertex_owners,
        source_map_id=target.geometry.source_map_id,
        axis_name="reanalysis_directory",
    )
    source_entry = CompositionEntry(
        target,
        entry_id="mesh/source",
        role="topology",
        owner_id="test-reanalysis",
        structure_id=target.mesh.mesh_id,
        revision_id=part.part_id,
        semantics_id=geometry_id,
    )
    part_entry = CompositionEntry(
        part,
        entry_id="mesh/part",
        role="topology",
        owner_id="test-reanalysis",
        structure_id=target.mesh.topology_id,
        revision_id=part.part_id,
        semantics_id=geometry_id,
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
            directory.composition_entry().binding("structure"),
        ),
    )
    directory = directory.bind_source_authority(
        part_entry.record_id, group, axis_name="reanalysis_directory"
    )
    composition = Composition(
        (source_entry, directory.composition_entry(), part_entry),
        boundary_id=manifest.manifest_id,
    )
    view = OwnerLocalCellGeometry.from_owner_local_result(
        distribution.owner_local_geometry_record(
            0,
            part_entry,
            composition,
            directory,
            source_id=part.name,
            source_revision=part.part_id,
            include_halo=False,
        )
    )
    local_mesh, local_geometry = view.reconstruct_local_geometry()
    assert local_mesh is not None and local_geometry is not None
    local_space = phx.discretization.FiniteElementPlan(
        local_mesh,
        tuple(layout.field_spec for layout in state.field_layouts),
        coordinate_spec=local_geometry,
    ).prepare()
    local_program = phx.equations.compile_finite_element_problem(
        target_program.form, local_space
    )
    plans = tuple(
        finite_element_dof_identity_plan(
            local_space,
            index,
            field_layout_id=layout.layout_id,
            distribution_id=view.distribution_id,
            topology_id=target.mesh.topology_id,
            geometry_id=geometry_id,
            local_rank=rank,
            part_count=1,
            cell_ids=local_cells,
            cell_owner_ranks=cell_owners,
            vertex_ids=vertices,
            vertex_owner_ranks=vertex_owners,
        )
        for index, layout in enumerate(state.field_layouts)
    )
    global_vertices = np.asarray(target.mesh.vertex_global_ids, dtype=np.int64)
    global_plans = tuple(
        finite_element_dof_identity_plan(
            target_all,
            index,
            field_layout_id=layout.layout_id,
            distribution_id=view.distribution_id,
            topology_id=target.mesh.topology_id,
            geometry_id=geometry_id,
            local_rank=rank,
            part_count=1,
            cell_ids=cell_ids,
            cell_owner_ranks=np.zeros(cell_ids.shape, dtype=np.int32),
            vertex_ids=global_vertices,
            vertex_owner_ranks=np.zeros(global_vertices.shape, dtype=np.int32),
        )
        for index, layout in enumerate(state.field_layouts)
    )
    sharded = []
    for global_plan, local_plan, value in zip(
        global_plans, plans, state.fields, strict=True
    ):
        global_rows = {
            tuple(key): row for row, key in enumerate(np.asarray(global_plan.dof_keys))
        }
        local_keys = tuple(tuple(key) for key in np.asarray(local_plan.dof_keys))
        assert set(global_rows) == set(local_keys)
        rows = np.asarray([global_rows[key] for key in local_keys], dtype=np.int64)
        sharded.append(local_plan.shard_values(np.asarray(value)[rows]))
    owner_state = phx.solver.FiniteElementAcceptedState(
        tuple(item[0] for item in sharded),
        state.time,
        state.step,
        state.topology_id,
        local_space.prepared_id,
        local_program.compilation_id,
        field_layouts=state.field_layouts,
        materials=state.materials,
        schedule_cursor=state.schedule_cursor,
        state_version=state.state_version,
        field_ownership=tuple(item[1] for item in sharded),
    )
    graph = DistributedWeightedCSRGraph(
        np.zeros(local_cells.size + 1, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        entity_ids=local_cells,
        owner_ranks=cell_owners,
        neighbor_owner_ranks=np.empty(0, dtype=np.int32),
        vertex_weights=np.ones(local_cells.shape, dtype=np.int64),
    )
    partition = partition_graph_distributed(
        graph,
        GraphPartitionPlan(1, maximum_imbalance=1.0),
        group,
        axis_name="reanalysis_partition",
    )
    migration = OwnerLocalGraphMigrationEvidence(
        partition,
        manifest.manifest_id,
        view.part_id,
        view.source_geometry_id,
        view.distribution_id,
        view.part_id,
        view.distribution_id,
        group.spec.group_id,
        rank,
        1,
        local_cells,
    )
    context = _FiniteElementRepartitionContext(
        source_manifest=phx.solver.FiniteElementRestartManifest(owner_state),
        source_compiled=local_program,
        target_compiled=local_program,
        source_plans=plans,
        target_plans=plans,
        migration=migration,
        source_composition=composition,
        execution_group=group,
        source_geometry_view=view,
        target_geometry_view=view,
        state_entry_id="solver/state",
        geometry_entry_id="mesh/geometry",
        field_ids=(),
        material_entry_id=None,
        auxiliary_ids=(),
        integrator_ids=(),
        target_auxiliary_layouts=None,
        target_integrator_layouts=None,
        target_material_layouts=None,
        source_material_plan=None,
        target_material_plan=None,
        retained=(),
        invalidated=(),
        reprepared=(),
        rank=rank,
        parts=1,
        axis_name="reanalysis_repartition",
    )
    _validate_finite_element_repartition_authority(context)
