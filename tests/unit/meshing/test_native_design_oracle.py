import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing.providers._native_implicit import _PreparedFixedImplicitRoute
from phydrax.meshing.providers._native_sources import NativeImplicitSource
from tools.meshing_qualification.scenarios.native_design import (
    _physical_oracle,
    _physical_solve,
    _sphere_source,
)


def test_physical_oracle_uses_each_block_of_the_single_field_dof_map() -> None:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    mesh = phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "upper", "triangle", np.array([[0, 1, 2]]), global_ids=(0,)
            ),
            phx.discretization.CellBlock(
                "lower", "triangle", np.array([[1, 0, 3]]), global_ids=(1,)
            ),
        ),
    )
    result = certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    # The shared edge has value one; the lower apex has value four.
    values = np.where(np.asarray(space.dof_maps[0].dof_coordinates)[:, 1] < 0, 4.0, 1.0)
    oracle = _physical_oracle(result, space, values, 1.0, 4)

    assert oracle["surface_area_m2"] == pytest.approx(1.0)
    assert oracle["represented_fe_qoi"] == pytest.approx(1.5)
    assert oracle["represented_exact_field_qoi"] == pytest.approx(1.0 / 6.0)


def test_fixed_lattice_publication_and_event_refusal() -> None:
    geometry, grid, source, plan, result, _ = _sphere_source(2)
    assert result.audit.passed
    assert result.certification is not None and result.certification.passed
    assert isinstance(plan.prepared, _PreparedFixedImplicitRoute)
    surface = plan.prepared.surface_plan
    index = geometry.schema.index(phx.geometry.ParameterId("design-sphere", "radius"))
    before = surface.realize(geometry.state)
    after = surface.realize(geometry.state.replace_at(index, np.asarray(0.68)))
    assert int(np.asarray(before.evidence.status)) == 0
    assert int(np.asarray(after.evidence.status)) & int(
        phx.geometry.ImplicitSurfaceStatus.SIGN_PATTERN_CHANGED
    )
    with pytest.raises(ValueError, match="stopped event"):
        _sphere_source(2, radius=0.68, fixed_plan=plan)
    with pytest.raises(ValueError, match="lattice"):
        _sphere_source(4, fixed_plan=plan)
    rebound_geometry, _, _, rebound_plan, rebound, _ = _sphere_source(
        2, radius=0.7557322279455441, fixed_plan=plan
    )
    assert isinstance(rebound_plan.prepared, _PreparedFixedImplicitRoute)
    assert rebound.result_id != result.result_id
    assert rebound_plan.prepared.surface_plan.plan_id == surface.plan_id
    expected = surface.realize(rebound_geometry.state)
    np.testing.assert_array_equal(rebound.mesh.coordinates, np.asarray(expected.vertices))
    qoi = _physical_solve(rebound, 0.7557322279455441)["physical_oracle"][
        "represented_fe_qoi"
    ]
    assert abs(qoi - 4.0 * np.pi * 0.75**4 / 3.0) <= 1.0e-8
    wrong_policy = phx.meshing.NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.3)
        ),
    )
    with pytest.raises(ValueError, match="policy"):
        phx.meshing.NativeMeshingProvider(wrong_policy).plan(
            source,
            plan.specification,
            coordinate_contract=plan.coordinate_contract,
            fixed_plan=plan,
        )
    wrong_geometry = phx.geometry.Box(
        (0.0, 0.0, 0.0), (1.5, 1.5, 1.5), feature_id="design-sphere"
    ).compile()
    wrong_source = NativeImplicitSource(
        wrong_geometry, source.source_id, "different-kernel-revision", grid=grid
    )
    wrong_scope = phx.meshing.MeshingScope(
        wrong_source.source_id,
        wrong_source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        wrong_source.scope_entity_set_id(2),
        wrong_source.scope_entity_ids(2),
    )
    control = plan.specification.size_controls[0]
    assert isinstance(control, phx.meshing.UniformSizeControl)
    wrong_specification = phx.meshing.SurfaceMeshingSpec(
        plan.specification.target,
        wrong_scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                wrong_scope, control.target_size, strength=control.strength
            ),
        ),
    )
    with pytest.raises(ValueError, match="kernel|schema"):
        phx.meshing.NativeMeshingProvider(plan.options).plan(
            wrong_source,
            wrong_specification,
            coordinate_contract=plan.coordinate_contract,
            fixed_plan=plan,
        )
    foreign_source = NativeImplicitSource(
        geometry, "other-sphere", source.source_revision, grid=grid
    )
    with pytest.raises((ValueError, phx.meshing.MeshingFailure)):
        phx.meshing.NativeMeshingProvider(plan.options).plan(
            foreign_source,
            plan.specification,
            coordinate_contract=plan.coordinate_contract,
            fixed_plan=plan,
        )


def test_surface_content_all_coefficients_and_reanalysis_rollback() -> None:
    _, _, _, _, source, _ = _sphere_source(2)
    solve = _physical_solve(source, 0.75)
    scope = phx.meshing.mesh_proposal_scope(source, 2)
    proposal = phx.meshing.MeshMarkingProposal(
        source, scope, np.ones(scope.entity_ids.size), proposer_id="conservative-test"
    )
    safety = phx.meshing.MeshProposalSafetyPolicy(
        source, minimum_size=0.04, maximum_size=0.35, maximum_displacement=0.0
    )
    proposal_transaction = phx.meshing.prepare_mesh_proposal(
        source,
        proposal,
        safety,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    adaptation = proposal_transaction.adaptation
    assert adaptation is not None
    target = proposal_transaction.commit(source)
    target_solve = _physical_solve(target, 0.75)
    element = phx.discretization.lagrange_element("triangle", 1)
    semantics = phx.solver.FiniteElementFieldTransferSemantics
    layouts = (
        phx.solver.FiniteElementAcceptedFieldLayout(
            phx.discretization.FiniteElementFieldSpec("u", element),
            transfer_semantics=semantics.NONCONSERVATIVE_INTERPOLATION,
        ),
        phx.solver.FiniteElementAcceptedFieldLayout(
            phx.discretization.FiniteElementFieldSpec("density", element),
            transfer_semantics=semantics.CONSERVATIVE_SURFACE_CONTENT,
        ),
    )
    field_specs = tuple(layout.field_spec for layout in layouts)
    source_all = phx.discretization.FiniteElementPlan(
        source.mesh, field_specs, coordinate_spec=source.geometry
    ).prepare()
    target_all = phx.discretization.FiniteElementPlan(
        target.mesh, field_specs, coordinate_spec=target.geometry
    ).prepare()
    full_form = phx.equations.FiniteElementForm(
        "native-design-surface-reaction", ("u", "density"), solve["problem"].form.actions
    )
    source_program = phx.equations.compile_finite_element_problem(full_form, source_all)
    target_program = phx.equations.compile_finite_element_problem(full_form, target_all)
    density = 1.0 + np.asarray(source.mesh.coordinates)[:, 0]
    accepted = phx.solver.FiniteElementAcceptedState(
        (solve["values"], density),
        0.0,
        0,
        source.mesh.topology_id,
        source_all.prepared_id,
        source_program.compilation_id,
        field_layouts=layouts,
    )
    executor = phx.solver.FiniteElementTopologyTransaction(
        lambda mesh, fields, materials, lineage, args: True
    )
    bad_solution = eqx.tree_at(
        lambda value: value.value,
        target_solve["solve_result"],
        jnp.zeros_like(target_solve["solve_result"].value),
    )
    bad = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        bad_solution,
        target_solve["solve_check"],
        "u",
        state_problem=target_program,
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    refused = executor.execute(
        accepted,
        source.mesh,
        adaptation,
        reanalysis_action=lambda *args: bad,
    )
    assert not bool(np.asarray(refused.committed))
    assert refused.state is accepted and refused.mesh is source.mesh
    good = phx.solver.FiniteElementReanalysisResult(
        target_solve["problem"],
        target_solve["solve_result"],
        target_solve["solve_check"],
        "u",
        state_problem=target_program,
        check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )
    committed = executor.execute(
        accepted,
        source.mesh,
        adaptation,
        reanalysis_action=lambda *args: good,
    )
    assert bool(np.asarray(committed.committed))
    (evidence,) = committed.surface_conservative_transfers
    assert evidence.transfer.conservative
    probe = np.linspace(-0.4, 0.8, len(density))
    assert np.dot(np.asarray(evidence.source_measures), probe) == pytest.approx(
        np.dot(
            np.asarray(evidence.target_measures),
            np.asarray(evidence.transfer.apply(probe)),
        ),
        abs=1.0e-11,
    )


def test_surface_content_refuses_nonaffine_restriction() -> None:
    from phydrax.discretization._cell_geometry import PerCellRestrictedCellGeometryElement
    from phydrax.discretization._coordinate_enclosure import axes, multiply
    from phydrax.solver._finite_element_adaptivity import _surface_p1_measures

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    block = mesh.blocks[0]
    cell_id = int(np.asarray(block.global_ids)[0])
    u, v = axes(2)
    element = PerCellRestrictedCellGeometryElement(
        phx.discretization.lagrange_element("triangle", 1),
        "triangle",
        (cell_id,),
        (cell_id,),
        ((multiply(u, u), v),),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {block.name: element},
        {block.name: np.asarray(((0, 1, 2),), dtype=np.int32)},
        mesh.coordinates,
    )
    with pytest.raises(ValueError, match="nonlinear or rational"):
        _surface_p1_measures(mesh, geometry)
