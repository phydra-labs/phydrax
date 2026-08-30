#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _tri_mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def test_finite_element_form_lowers_to_typed_actions_and_worksets():
    mesh = _tri_mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.FiniteElementForm(
        "poisson-ir",
        "u",
        (phx.equations.DiffusionAction("u"), phx.equations.SourceAction("u", 1.0)),
    )

    action_ir = phx.equations.fem.lower_finite_element_form(form, discretization)
    workset_program = phx.equations.fem.compile_workset_program(
        action_ir, form, discretization
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    assert action_ir.ir_id
    assert tuple(slot.name for slot in action_ir.slots) == ("u",)
    assert len(action_ir.actions) == 2
    assert workset_program.program_id
    assert all(workset.entity_indices.size for workset in workset_program.worksets)
    assert len(workset_program.worksets) == 2
    assert sum(workset.action_indices.size for workset in workset_program.worksets) == 2
    assert compiled._kernel_table.table_id
    assert compiled._workset_program.program_id == workset_program.program_id


def test_high_order_tensor_family_partition_unity_and_sum_factorization():
    family = phx.discretization.fem.ReferenceNodalFamily(
        "quadrilateral", 3, node_set="gauss-lobatto"
    )
    points = jnp.asarray([[0.2, 0.3], [0.7, 0.8]])
    element = family.finite_element()
    values, gradients = element.tabulate(points)
    tabulation = phx.discretization.fem.TensorProductTabulation(
        family,
        (jnp.asarray([0.2, 0.7]), jnp.asarray([0.3, 0.8])),
    )
    plan = phx.discretization.fem.SumFactorizationPlan(tabulation)
    coefficients = jnp.arange(16.0).reshape((4, 4))
    interpolated = plan.interpolate(coefficients)

    assert jnp.allclose(jnp.sum(values, axis=-1), 1.0)
    assert jnp.allclose(jnp.sum(gradients, axis=1), 0.0, atol=1.0e-12)
    assert interpolated.shape == (2, 2)
    assert phx.discretization.fem.QuadratureChunkPolicy(3).chunks(8) == (
        (0, 3),
        (3, 6),
        (6, 8),
    )


def test_edge_and_node_smoothing_partition_patch_and_rigid_modes():
    mesh = _tri_mesh()
    smoothing = phx.discretization.fem.smoothing
    constitutive = smoothing.plane_stress_matrix(1.0, 0.3)
    edge = smoothing.SmoothedElasticityPlan("ES", mesh, constitutive)
    node = smoothing.SmoothedElasticityPlan("NS", mesh, constitutive)
    edge_geometry = edge.geometry(mesh.coordinates)
    node_geometry = node.geometry(mesh.coordinates)
    edge_stiffness = edge.operator(mesh.coordinates).materialize()
    node_stiffness = node.operator(mesh.coordinates).materialize()
    translation_x = jnp.tile(jnp.asarray([1.0, 0.0]), (5, 1)).reshape((-1,))
    rotation = jnp.stack(
        (-mesh.coordinates[:, 1], mesh.coordinates[:, 0]), axis=-1
    ).reshape((-1,))

    assert jnp.allclose(jnp.sum(edge_geometry.area), 1.0)
    assert jnp.allclose(jnp.sum(node_geometry.area), 1.0)
    assert jnp.allclose(edge_stiffness, edge_stiffness.T)
    assert jnp.allclose(node_stiffness, node_stiffness.T)
    assert jnp.linalg.norm(edge_stiffness @ translation_x) < 1.0e-10
    assert jnp.linalg.norm(node_stiffness @ translation_x) < 1.0e-10
    assert jnp.linalg.norm(edge_stiffness @ rotation) < 1.0e-10
    assert jnp.linalg.norm(node_stiffness @ rotation) < 1.0e-10


def test_q4_plate_smoothing_keeps_channel_partitions_independent():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    block = phx.discretization.CellBlock(
        "quads", "quadrilateral", jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    )
    mesh = phx.discretization.CellMesh(vertices, (block,))
    plan = phx.discretization.fem.smoothing.Q4FSDTSmoothingPlan(mesh)
    channels = plan.channels(vertices)

    assert channels.membrane_gradient.shape == (3, 4, 2)
    assert channels.bending_gradient.shape == (3, 4, 2)
    assert channels.shear_average.shape == (1, 4)
    assert channels.nonlinear_gradient.shape == (3, 4, 2)


def test_fully_smoothed_axisymmetric_stiffness_and_mass_are_symmetric():
    vertices = jnp.asarray([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    block = phx.discretization.CellBlock(
        "quads", "quadrilateral", jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    )
    mesh = phx.discretization.CellMesh(vertices, (block,))
    plan = phx.discretization.fem.smoothing.FullySmoothedAxisymmetricPlan(
        "CS", mesh, jnp.eye(4)
    )
    primitive = plan.layout.boundary_shape_values
    smoothed_shape = jnp.mean(plan.layout.boundary_shape_values, axis=(1, 2))
    stiffness, mass, geometry = plan.operators(vertices, primitive, smoothed_shape)

    assert jnp.all(geometry.centroid[:, 0] > 0.0)
    assert jnp.allclose(stiffness, jnp.swapaxes(stiffness, -1, -2))
    assert jnp.allclose(mass, jnp.swapaxes(mass, -1, -2))
    assert jnp.all(jnp.linalg.eigvalsh(mass) >= -1.0e-12)


def test_local_implicit_material_uses_implicit_jvp():
    material = phx.equations.fem.LocalImplicitMaterial(
        lambda state, target: state**2 - target,
        lambda state, target: phx.equations.ConstitutiveResponse(state, state),
        state_shape=(1,),
        model_id="square-root",
    )
    initial = jnp.asarray([1.0])
    root = material.solve(initial, jnp.asarray([4.0]))
    _, tangent = jax.jvp(
        lambda target: material.solve(initial, target),
        (jnp.asarray([4.0]),),
        (jnp.asarray([1.0]),),
    )

    assert jnp.allclose(root, 2.0, atol=1.0e-10)
    assert jnp.allclose(tangent, 0.25, atol=1.0e-8)


def test_time_law_schedule_and_uniform_refinement_are_transactional():
    law = phx.solver.TimeLaw.ramp(0.0, 1.0, 0.0, 1.0)

    def solve(state, start, end, time_law, args):
        return phx.solver.ScheduleStepResult(
            state=state + time_law.value(end),
            accepted=jnp.asarray(True),
            diagnostics=jnp.asarray(0.0),
        )

    schedule = phx.solver.SolveSchedule(
        (phx.solver.SolveStage("load", 0.0, 1.0, law, solve),)
    )
    final_state, results = schedule.run(jnp.asarray(0.0))
    refined, refinement = phx.discretization.fem.refine_triangles_uniform(_tri_mesh())

    assert jnp.allclose(final_state, 1.0)
    assert bool(results[0].accepted)
    assert refined.blocks[0].cell_count == 16
    assert refinement.child_cells.shape == (4, 4)


def test_element_partial_and_p_transfer_operators_are_consistent():
    local_matrix = jnp.asarray([[[2.0, -1.0], [-1.0, 2.0]], [[2.0, -1.0], [-1.0, 2.0]]])
    gathers = jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32)
    element = phx.equations.fem.ElementTensorOperator(
        local_matrix, gathers, gathers, 3, 3
    )
    value = jnp.asarray([1.0, 2.0, 3.0])
    expected = jnp.asarray([0.0, 4.0, 4.0])
    assert jnp.allclose(element.mv(value), expected)
    assert jnp.allclose(element.diagonal(), jnp.asarray([2.0, 4.0, 2.0]))
    assert jnp.allclose(element.transpose_mv(value), expected)
    assert jnp.allclose(element.as_sparse_coordinate().mv(value), expected)

    basis = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    partial = phx.equations.fem.PartialAssemblyOperator(
        basis,
        jnp.ones((2, 2)),
        jnp.ones((2, 2)),
        gathers,
        3,
    )
    assert partial.mv(value).shape == value.shape
    assert jnp.allclose(
        partial.as_element_tensor().mv(value),
        partial.mv(value),
    )
    assert jnp.allclose(
        partial.as_sparse_coordinate().mv(value),
        partial.mv(value),
    )

    coarse = phx.discretization.fem.ReferenceNodalFamily("quadrilateral", 1)
    fine = phx.discretization.fem.ReferenceNodalFamily("quadrilateral", 3)
    transfer = phx.discretization.fem.quadrilateral_p_transfer(coarse, fine)
    constant = jnp.ones((4,))
    assert jnp.allclose(transfer.prolong(constant), 1.0)


def test_application_model_primitives_are_executable():
    cpfem = phx.applications.crystal_plasticity
    material = cpfem.CrystalPlasticityModel(
        (
            cpfem.CrystalSlipSystem(
                jnp.asarray([1.0, 0.0, 0.0]),
                jnp.asarray([0.0, 1.0, 0.0]),
            ),
        ),
        cpfem.CrystalPlasticityParameters(1.0, 2.0, 0.01, 0.1, 0.2, 1.0),
    )
    response = material.update(jnp.eye(3), material.initial_state(), 0.1)
    fracture = phx.applications.fracture.PhaseFieldFractureParameters(1.0, 1.0, 1.0, 0.1)
    contact = phx.applications.contact.FrictionlessContactLaw(100.0)
    traction, tangent, active = contact.response(
        jnp.asarray(-0.01), jnp.asarray([1.0, 0.0])
    )

    assert response.state.plastic_deformation.shape == (3, 3)
    assert fracture.degradation(jnp.asarray(0.0)) > 0.0
    assert jnp.allclose(traction, jnp.asarray([1.0, 0.0]))
    assert tangent > 0.0
    assert active


def test_partition_and_local_adaptation_have_stable_routes():
    mesh = _tri_mesh()
    partition = phx.discretization.fem.partition_cells_contiguous(mesh, 2)
    marked = phx.discretization.fem.maximum_mark(
        jnp.asarray([4.0, 1.0, 1.0, 1.0]),
        0.25,
        cell_global_ids=mesh.blocks[0].global_ids,
    )
    refined, adaptation, transfer = phx.discretization.fem.refine_triangles_local(
        mesh, marked
    )

    assert set(partition.cell_owner.tolist()) == {0, 1}
    assert adaptation.parent_cell_ids.shape == (1,)
    assert transfer.primal.shape == (
        refined.coordinates.shape[0],
        mesh.coordinates.shape[0],
    )
    assert refined.blocks[0].cell_count == 5
